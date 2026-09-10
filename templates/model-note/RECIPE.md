# Recipe — &lt;best tok/s&gt;

Checkpoint: `org/name`  
Hardware: 1× Gaudi2 96 GB (change if not).

```bash
export PT_HPU_LAZY_MODE=0
# …

vllm serve /path/to/checkpoint \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Mark any **out-of-tree** patch in a separate section so a stock checkout is obvious.

## Do not

Flags that look tempting and fail on this model.
