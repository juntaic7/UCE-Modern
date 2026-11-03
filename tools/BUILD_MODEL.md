# Build the Hugging Face model repos

```bash
# Build locally into converted/<model>/
python tools/build_hub_repo.py --model uce-100m
python tools/build_hub_repo.py --model uce-650m

# Build and push to the Hub
python tools/build_hub_repo.py --model uce-100m --push <hf-user>/UCE-Modern-100m
python tools/build_hub_repo.py --model uce-650m --push <hf-user>/UCE-Modern-650m
```
