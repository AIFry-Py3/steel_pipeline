Setup:
1. Install Ollama and make sure it's running.
2. Run ollama pull qwen3-vl:8b-instruct.
3. Run pip install ollama pymupdf python-dotenv.
4. Test with $env:MAX_FILES=1

`100% CPU` confirms it. Ollama isn't using your laptop's GPU for this model, so the 8B model runs on the processor alone. That explains the slow "say hi" and the 600s timeout. Your script sends 3 page images with a context of 16384, which is far heavier than the 4096 context in this test.

**Next step:** run the full job on the college machine. Check there first, before running the script:
1. Run `nvidia-smi` and note the GPU name and VRAM.
2. Run `ollama run qwen3-vl:8b-instruct "say hi"`. It should answer in a few seconds.
3. Run `ollama ps` and confirm `PROCESSOR` shows `100% GPU`.

If step 3 shows CPU or a split there too, the VRAM is probably too small for the model. Then use `qwen3-vl:4b` after `ollama pull qwen3-vl:4b`, which is the fallback your log said wasn't installed.

On the laptop, don't run the full pipeline, because it will time out.

Before you run step 3 on the college machine, what would you expect `PROCESSOR` to show if the GPU has 8 GB of VRAM and the model plus context needs more than that?