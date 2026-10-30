"""Deterministic virtual-AO probe for the 50 ms missing-target guard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import echoChamber as app


def main() -> int:
    config = app.AppConfig(processing_block_ms=5.0, ao_target_lead_ms=100.0)
    state = app.RuntimeState(
        esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
        pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
        controller_ready={"esn": False, "rule": True, "pid": True},
    )
    state.set_stim("threshold_pulse", 0.25)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    scheduler = app.SimulatedAoScheduler(config, state)

    supplied = []
    for source_block in range(10):
        target = source_block + config.ao_lead_chunks
        assert scheduler.submit(target, np.zeros(config.chunk_size), state.snapshot().generation)
        supplied.append({"source_block": source_block, "target_block": target})
        scheduler.advance_elapsed(config.chunk_size / config.sample_rate)
        assert state.snapshot().fault is None

    delayed_source_block = 10
    target = delayed_source_block + config.ao_lead_chunks
    timeline = []
    for tick in range(1, 12):
        # advance_elapsed consumes one 5 ms hardware-clock block, then services
        # the host writer once. Compute exactly the pre-refill decision inputs.
        generated_before = scheduler.generated_sample_count
        generated_at_service = generated_before + config.chunk_size
        next_write = scheduler._next_write_block
        queued = next_write * config.chunk_size - generated_at_service
        deficit = max(0, int(np.ceil(
            (config.ao_lead_samples - queued) / config.chunk_size
        )))
        snapshot = state.snapshot()
        ready = 0
        for candidate in range(next_write, next_write + deficit):
            pending = scheduler.commands.get(candidate)
            if pending is None or not scheduler.lifecycle.eligible(
                    pending, snapshot.generation, snapshot.transition_pending):
                break
            ready += 1
        missing = deficit - ready
        first_missing = next_write + ready if missing else None
        headroom = first_missing * config.chunk_size - generated_at_service if missing else None
        urgent = bool(missing and scheduler.lifecycle.missing_refill_is_urgent(
            next_write, ready, generated_at_service, config.chunk_size,
            config.ao_deadline_samples,
        ))
        pre = {
            "tick": tick,
            "generated_before_advance": generated_before,
            "generated_at_pre_refill": generated_at_service,
            "next_write_block_pre_refill": next_write,
            "queued_samples_pre_refill": queued,
            "deficit_blocks": deficit,
            "ready_contiguous_prefix_blocks": ready,
            "missing_blocks": missing,
            "first_missing_target_block": first_missing,
            "first_missing_target_sample_index": (
                first_missing * config.chunk_size if first_missing is not None else None
            ),
            "first_missing_target_headroom_samples": headroom,
            "first_missing_target_headroom_ms": (
                headroom * 1000.0 / config.sample_rate if headroom is not None else None
            ),
            "urgent_missing": urgent,
        }
        generated = scheduler.advance_elapsed(config.chunk_size / config.sample_rate)
        after = scheduler.stats_snapshot()
        pre["generated_after_service"] = scheduler.generated_sample_count
        pre["next_write_block_after_service"] = scheduler._next_write_block
        pre["active_demand_misses_after_service"] = after["active_demand_misses"]
        pre["zero_substitutions_after_service"] = after["zero_substitutions"]
        pre["fault_after_service"] = state.snapshot().fault
        pre["generated_output_blocks"] = [
            {"sample_index": block.sample_index, "stage": block.stage,
             "nonzero_samples": int(np.count_nonzero(block.waveform_v))}
            for block in generated
        ]
        timeline.append(pre)
        if state.snapshot().fault:
            break

    event = timeline[-1]
    assert event["first_missing_target_block"] == target
    assert event["first_missing_target_headroom_samples"] == config.ao_deadline_samples
    assert event["urgent_missing"] is True
    assert event["active_demand_misses_after_service"] == 1
    assert timeline[-2]["first_missing_target_headroom_samples"] > config.ao_deadline_samples
    assert timeline[-2]["urgent_missing"] is False
    result = {
        "schema": "echo-chamber-deterministic-ao-deadline-probe-v1",
        "purpose": "Prove the simulated scheduler's missing-target guard boundary only.",
        "sample_rate_hz": config.sample_rate,
        "chunk_samples": config.chunk_size,
        "ao_lead_samples": config.ao_lead_samples,
        "ao_lead_ms": config.ao_target_lead_ms,
        "ao_deadline_samples": config.ao_deadline_samples,
        "ao_deadline_ms": config.ao_deadline_samples * 1000.0 / config.sample_rate,
        "supplied_source_blocks": supplied,
        "delayed_source_block": delayed_source_block,
        "delayed_source_target_block": target,
        "producer_delay_ticks": len(timeline),
        "pre_refill_timeline": timeline,
        "final_scheduler_stats": scheduler.stats_snapshot(),
        "final_fault": state.snapshot().fault,
        "interpretation_limit": (
            "Deterministic simulator behavior, not paced host timing, actual NI underflow, "
            "physical output, or a cause attribution for an observed run."
        ),
    }
    output = Path(__file__).resolve().parent / "diagnostics" / "ao-deadline-boundary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(output)
    print(json.dumps({
        "pre_threshold_headroom_samples": timeline[-2]["first_missing_target_headroom_samples"],
        "fault_boundary_headroom_samples": event["first_missing_target_headroom_samples"],
        "deadline_samples": config.ao_deadline_samples,
        "missing_target_block": event["first_missing_target_block"],
        "active_demand_misses": event["active_demand_misses_after_service"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
