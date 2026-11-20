# Disaggregated Serving

This example contains scripts that demonstrate the disaggregated serving features of vLLM.

## Files

- `disagg_proxy_demo.py` - Demonstrates XpYd (X prefill instances, Y decode instances).
- `disagg_proxy_overload_relief.py` - Demonstrates sending requests straight to the decode instance while the prefill instance is backed up.
- `kv_events.sh` - Demonstrates KV cache event publishing.
- `mooncake_connector` - A proxy demo for MooncakeConnector.
