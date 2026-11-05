# Gemma serial scoring amendment

The owner approved serial Gemma scoring after both
workers failed batched numerical parity. Worker 00 reported maximum absolute
differences of 0.1796875 for logits and 0.25 for hidden states. Worker 01
reported 0.1015625 and 0.1875. The preregistered limit is 0.02.

This amendment supersedes the batching and out-of-memory fallback rules
for Gemma. Each pass uses one fresh FP16 forward on a T4 with prefix caching
disabled. Each worker checks repeatability on ten full passes and on every
required pass for five calibration questions. The 0.02 limits, finite-value
checks, identical answers and ten generation agreements apply to this smoke.
Its outcome is serial repeatability. The previous batched parity failures
remain part of the run record.

The owner times the same 50 stratified calibration questions in serial mode.
The owner run can proceed automatically after the serial smoke passes on
both workers and the six-worker planning forecast is at most 2.5 hours.
The dataset version, pass set and sealed test procedure follow the original
preregistration. The run identity binds this amendment to the worker code.
