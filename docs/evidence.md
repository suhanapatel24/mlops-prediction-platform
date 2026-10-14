# MLOps Evidence

Implemented:
- [x] reproducible training code
- [x] version-pinned dependencies
- [x] evaluation metadata contract
- [x] serialized preprocessing + estimator
- [x] FastAPI inference service
- [x] input validation
- [x] automated tests
- [x] multi-stage Docker build
- [x] CI training and metric gate
- [x] Kubernetes deployment definition

Verify before resume claims:
- [ ] GitHub Actions run passes
- [ ] actual training metrics captured from CI
- [ ] container runs and /healthz responds
- [ ] /predict returns a valid response
- [ ] image published to a registry
- [ ] Kubernetes deployment becomes Ready
- [ ] LoadBalancer endpoint tested

Do not claim deployment, latency, uptime, or model-performance results until those measurements are actually collected.
