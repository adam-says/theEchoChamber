#!/usr/bin/env python3
"""Run a short core smoke, the nine-case replay matrix, or the paced soak."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarkControllers import (  # noqa: E402
    CONTROLLERS, DEFAULT_TRACES, LEAD_SWEEP_MS, acceptance_failures, json_safe,
    _browser_connection_url, run_replay_case, write_progress, write_results,
)
from simulation import (  # noqa: E402
    TRACKING_PID_SETTINGS, TRACKING_PLANT_SETTINGS, run_feedback_comparison,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run 1 second from every trace/controller pair")
    mode.add_argument("--matrix", action="store_true", help="run all three full finite traces with all controllers")
    mode.add_argument("--paced", action="store_true", help="pace a real-time 30 minute run for each controller")
    mode.add_argument("--simulation", action="store_true", help="compare responsive plant feedback with stimulation disabled")
    mode.add_argument("--tracking-simulation", action="store_true",
                      help="run the reachable PID target-tracking synthetic scenario")
    p.add_argument("--lead-sweep", action="store_true", help="repeat the selected mode at every selected AO lead")
    p.add_argument("--trace", type=Path, action="append", help="trace path; repeat to override the three canonical defaults")
    p.add_argument("--controller", choices=CONTROLLERS, action="append", help="controller selection; repeat to select more than one")
    p.add_argument("--lead-ms", type=float, default=100.0, help="AO target lead for smoke, matrix, and simulation runs")
    p.add_argument("--leads-ms", type=float, nargs="+", default=list(LEAD_SWEEP_MS), help="lead values used by --lead-sweep")
    p.add_argument("--duration-seconds", type=float, help="finite source duration override; default is 1s for smoke and full trace for matrix")
    p.add_argument("--paced-seconds", type=float, default=1800.0, help="real elapsed duration for each --paced controller run")
    p.add_argument("--simulation-seconds", type=float, default=24.0, help="synthetic plant duration for --simulation")
    p.add_argument("--tracking-simulation-seconds", type=float, default=40.0,
                   help="duration for the reachable PID target-tracking case")
    p.add_argument("--output-dir", type=Path, default=ROOT / "work" / "validation_runs")
    p.add_argument("--progress-log", type=Path, help="JSON Lines progress file; defaults inside --output-dir")
    p.add_argument("--progress-interval-seconds", type=float, default=10.0)
    p.add_argument("--inject-stall-ms", type=float, default=0.0,
                   help="inject one producer stall (must be longer than AO lead to test underrun handling)")
    p.add_argument("--stall-at-second", type=float,
                   help="finite replay source time at which to inject the stall")
    p.add_argument("--keep-recordings", action="store_true", help="retain the HDF5 recorder file for each case")
    browser = p.add_mutually_exclusive_group()
    browser.add_argument("--browser-ui", action="store_true", help="open index.html automatically and require its WebSocket connection")
    browser.add_argument("--external-browser", action="store_true",
                          help="emit browser_required progress and wait for an externally opened UI")
    p.add_argument("--browser-base-url", default="http://127.0.0.1:8000",
                   help="static UI server origin or index.html URL used with --external-browser")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.progress_interval_seconds <= 0:
        raise SystemExit("--progress-interval-seconds must be positive")
    if args.inject_stall_ms < 0 or (args.inject_stall_ms and args.inject_stall_ms <= args.lead_ms):
        raise SystemExit("--inject-stall-ms must be zero or greater than --lead-ms")
    if args.inject_stall_ms > 0 and not (args.simulation or args.tracking_simulation) and args.stall_at_second is None:
        raise SystemExit("finite replay stall injection requires --stall-at-second")
    if args.stall_at_second is not None and (args.stall_at_second < 0 or args.inject_stall_ms <= 0):
        raise SystemExit("--stall-at-second requires a positive --inject-stall-ms")
    if args.paced and args.paced_seconds <= 0:
        raise SystemExit("--paced-seconds must be positive")
    if args.paced and not (args.browser_ui or args.external_browser):
        raise SystemExit("--paced requires --browser-ui or --external-browser so the real controller UI is connected")
    if args.browser_ui or args.external_browser:
        try:
            _browser_connection_url(args.browser_base_url, 1)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.progress_log.resolve() if args.progress_log else output_dir / "progress.jsonl"
    traces = [x.resolve() for x in args.trace] if args.trace else list(DEFAULT_TRACES)
    controllers = args.controller or list(CONTROLLERS)
    record_dir = output_dir / "recordings" if args.keep_recordings else None
    start = {
        "type": "validation_started", "pid": os.getpid(),
        "mode": next(name for name in ("smoke", "matrix", "paced", "simulation", "tracking_simulation", "lead_sweep") if getattr(args, name)),
        "command_line": sys.argv, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "browser_mode": ("external" if args.external_browser else "automatic" if args.browser_ui else "not_required"),
        "browser_base_url": args.browser_base_url if args.external_browser else None,
    }
    write_progress(progress, start)
    metrics = []
    environments = []
    simulations = []
    if args.simulation or args.tracking_simulation:
        selected = controllers if args.simulation else (args.controller or ["pid"])
        duration = (args.duration_seconds or args.simulation_seconds) if args.simulation else (
            args.duration_seconds or args.tracking_simulation_seconds)
        plant_settings = None
        pid_settings = None
        if args.tracking_simulation:
            plant_settings = replace(TRACKING_PLANT_SETTINGS,
                                     disturbance_stop_s=min(TRACKING_PLANT_SETTINGS.disturbance_stop_s,
                                                            duration - 1.0))
            pid_settings = TRACKING_PID_SETTINGS
            if plant_settings.disturbance_stop_s <= plant_settings.disturbance_start_s + 2.0:
                raise SystemExit("tracking simulation duration must leave at least two disturbance seconds after calibration")
        for controller in selected:
            simulations.append(run_feedback_comparison(
                controller, duration_s=duration,
                lead_ms=args.lead_ms, progress_path=progress,
                recording_dir=record_dir, paced=True,
                plant_settings=plant_settings, pid_settings=pid_settings,
                inject_stall_ms=args.inject_stall_ms,
                stall_on_first_pulse=(args.inject_stall_ms > 0 and not args.tracking_simulation),
            ))
        sim_path = output_dir / f"responsive_simulation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        sim_path.write_text(json.dumps(json_safe({
            "schema": "echo-chamber-responsive-simulation-matrix-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "physical_pulse_claims": False, "cases": simulations,
            "scenario": "reachable_target_tracking" if args.tracking_simulation else "saturation_and_activity_comparison",
        }), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        csv_path = sim_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            fields = ("controller", "stimulated_disturbance_rms_mv", "no_stimulation_disturbance_rms_mv",
                      "observed_disturbance_rms_reduction_mv", "max_requested_rate_hz",
                      "rendered_pulse_events", "pulse_commit_samples", "generated_nonzero_ao_samples",
                      "stimulated_iae_z_seconds", "control_iae_z_seconds",
                      "tracking_iae_improvement_z_seconds", "mean_rate_hz", "rate_std_hz",
                      "saturation_fraction", "settling_time_seconds", "rate_direction_reversals",
                      "tracking_window_complete", "tracking_coverage_fraction",
                      "physical_pulse_claims")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for case in simulations:
                writer.writerow({
                    "controller": case["controller"],
                    "stimulated_disturbance_rms_mv": case["stimulated"]["activity"]["disturbance_ca3_block_rms_mv"],
                    "no_stimulation_disturbance_rms_mv": case["no_stimulation_control"]["activity"]["disturbance_ca3_block_rms_mv"],
                    "observed_disturbance_rms_reduction_mv": case["observed_disturbance_rms_reduction_mv"],
                    "max_requested_rate_hz": case["stimulated"]["metrics"]["max_requested_rate_hz"],
                    "rendered_pulse_events": case["stimulated"]["metrics"]["rendered_pulse_events"],
                    "pulse_commit_samples": case["stimulated"]["metrics"]["pulse_commit_samples"],
                    "generated_nonzero_ao_samples": case["accepted_generated_command_samples"],
                    "stimulated_iae_z_seconds": case["stimulated"]["controller_tracking"]["iae_z_seconds"],
                    "control_iae_z_seconds": case["no_stimulation_control"]["controller_tracking"]["iae_z_seconds"],
                    "tracking_iae_improvement_z_seconds": case["tracking_error_reduction_z_seconds"],
                    "mean_rate_hz": case["stimulated"]["controller_tracking"]["mean_rate_hz"],
                    "rate_std_hz": case["stimulated"]["controller_tracking"]["rate_std_hz"],
                    "saturation_fraction": case["stimulated"]["controller_tracking"]["saturation_fraction"],
                    "settling_time_seconds": case["stimulated"]["controller_tracking"]["settling_time_seconds"],
                    "rate_direction_reversals": case["stimulated"]["controller_tracking"]["rate_direction_reversals"],
                    "tracking_window_complete": case["stimulated"]["controller_tracking"]["window_complete"],
                    "tracking_coverage_fraction": case["stimulated"]["controller_tracking"]["coverage_fraction"],
                    "physical_pulse_claims": False,
                })
        for item in simulations:
            change = item["observed_disturbance_rms_reduction_mv"]
            change_text = f"{change:.6f} mV" if change is not None else "unavailable for this duration"
            print(f"{item['controller']}: observed change={change_text}, "
                  f"generated AO samples={item['accepted_generated_command_samples']}")
        print(f"JSON: {sim_path}\nCSV: {csv_path}\nProgress: {progress}")
        write_progress(progress, {"type": "validation_complete", "result": str(sim_path),
                                  "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return 0

    leads = args.leads_ms if args.lead_sweep else [args.lead_ms]
    if args.lead_sweep and args.duration_seconds is None and not args.smoke:
        print("Lead sweep uses full traces unless --duration-seconds limits each run.")
    if args.smoke:
        duration = args.duration_seconds if args.duration_seconds is not None else 1.0
    elif args.paced:
        duration = args.duration_seconds if args.duration_seconds is not None else args.paced_seconds
        if not args.trace:
            traces = traces[:1]
    else:
        duration = args.duration_seconds
    for lead in leads:
        for trace in traces:
            for controller in controllers:
                print(f"start controller={controller} trace={trace.name} lead_ms={lead:g}", flush=True)
                result, environment = run_replay_case(
                    trace, controller, lead_ms=lead, limit_seconds=duration,
                    paced=args.paced, progress_path=progress,
                    progress_interval_s=args.progress_interval_seconds,
                    recording_dir=record_dir,
                    browser_ui=args.browser_ui or args.external_browser,
                    external_browser=args.external_browser,
                    browser_base_url=args.browser_base_url,
                    expected_source_samples=(36_000_000 if not args.trace else None),
                    inject_stall_ms=args.inject_stall_ms,
                    stall_at_seconds=args.stall_at_second,
                )
                metrics.append(result)
                environments.append(environment)
                print(f"done case={result.case_id} blocks={result.processed_blocks} "
                      f"p99={result.process_p99_ms:.3f}ms late={result.deadline_exceeded_blocks}", flush=True)
    run_kind = "smoke" if args.smoke else "paced" if args.paced else "lead_sweep" if args.lead_sweep else "matrix"
    gate_enabled = bool(args.matrix or args.paced)
    acceptance = [
        {"case_id": result.case_id, "passed": not acceptance_failures(result),
         "failures": acceptance_failures(result)}
        for result in metrics
    ] if gate_enabled else []
    json_path, csv_path = write_results(metrics, environments, output_dir, run_kind=run_kind,
                                        acceptance=acceptance)
    write_progress(progress, {"type": "validation_complete", "pid": os.getpid(),
                              "json": str(json_path), "csv": str(csv_path),
                              "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"JSON: {json_path}\nCSV: {csv_path}\nProgress: {progress}")
    if gate_enabled and any(not item["passed"] for item in acceptance):
        failed = sum(not item["passed"] for item in acceptance)
        print(f"ACCEPTANCE FAILED: {failed}/{len(acceptance)} cases; results retained at {json_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
