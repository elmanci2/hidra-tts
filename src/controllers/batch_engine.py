import uuid
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class BatchJob:
    """A single TTS request waiting in the queue."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    audio_ref_path: Optional[str] = None
    voice_clone_prompt_bytes: Optional[bytes] = None
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class BatchEngine:
    """
    Dynamic Batching Engine for TTS inference.

    Architecture:
    1. Incoming HTTP requests are pushed into an asyncio.Queue as BatchJob items.
    2. A background coroutine (the collector) continuously drains the queue.
    3. It groups jobs into batches sized by available VRAM.
    4. Each batch is sent to the GPU in a SINGLE forward pass (true batched inference).
    5. Results are delivered back to each waiting HTTP request via asyncio.Future.

    This means:
    - The GPU never sits idle waiting for I/O.
    - Multiple requests are fused into one matrix operation.
    - VRAM is never exceeded because batch size is dynamically calculated.
    """

    def __init__(self, generator, max_batch_size: int = 8,
                 max_wait_ms: float = 200, vram_per_item_gb: float = 0.5):
        self.generator = generator
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.vram_per_item_gb = vram_per_item_gb
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self.total_batches_processed = 0
        self.total_items_processed = 0

    def start(self):
        """Start the background batch collector loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._batch_loop())
            print("🔄 Batch Engine started.")

    def stop(self):
        """Stop the background batch collector loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def submit(self, text: str, audio_ref_path: Optional[str] = None,
                     voice_clone_prompt_bytes: Optional[bytes] = None,
                     **gen_kwargs) -> bytes:
        """
        Submit a single TTS request. Returns the WAV bytes when the batch
        containing this request has been processed.
        """
        job = BatchJob(
            text=text,
            audio_ref_path=audio_ref_path,
            voice_clone_prompt_bytes=voice_clone_prompt_bytes,
            gen_kwargs=gen_kwargs,
            future=asyncio.get_event_loop().create_future(),
        )
        await self.queue.put(job)
        return await job.future

    def _calculate_dynamic_batch_size(self) -> int:
        """Calculate max batch size based on the initialized max_batch_size."""
        # Aplicamos la reducción del 5% directamente sobre el tamaño máximo
        # estático que se pasó al inicializar el engine, para evitar que
        # el re-caché de PyTorch reduzca el tamaño del batch a 1.
        dynamic_max = max(1, int(self.max_batch_size * 0.95))
        return dynamic_max

    async def _batch_loop(self):
        """
        Main batch collector loop.

        Strategy:
        - Wait for at least 1 item in the queue.
        - Then collect more items for up to max_wait_ms.
        - Once we have a batch (or timeout), process it.
        """
        print(f"⚡ Batch loop running (max_batch={self.max_batch_size}, max_wait={self.max_wait_ms*1000}ms)")
        while self._running:
            try:
                # Wait for the first item (blocks until something arrives)
                first_job = await self.queue.get()
                batch: List[BatchJob] = [first_job]

                # Calculate how many more we can fit
                dynamic_max = self._calculate_dynamic_batch_size()

                # Wait until the batch is full OR the timeout expires
                while len(batch) < dynamic_max:
                    try:
                        # Use wait_for to implement the temporal window
                        job = await asyncio.wait_for(self.queue.get(), timeout=self.max_wait_ms)
                        batch.append(job)
                        print(f"📥 Item añadido al batch ({len(batch)}/{dynamic_max})...")
                    except asyncio.TimeoutError:
                        # If no more items arrive within the window, stop waiting and process what we have
                        if len(batch) > 0:
                            print(f"⏱️ Tiempo de espera agotado ({self.max_wait_ms*1000}ms). Procesando batch incompleto de {len(batch)}.")
                        break

                # Process the batch
                await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Batch loop error: {e}")
                # Fail all pending jobs in this batch
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(e)

    async def _process_batch(self, batch: List[BatchJob]):
        """Process a collected batch through the GPU."""
        texts = [job.text for job in batch]
        batch_size = len(texts)

        # All jobs in a batch must share the same voice reference.
        # We use the first job's reference for the entire batch.
        ref_path = batch[0].audio_ref_path
        prompt_bytes = batch[0].voice_clone_prompt_bytes
        gen_kwargs = batch[0].gen_kwargs.copy()

        start_time = time.time()
        print(f"🧠 Processing batch of {batch_size} items...")

        try:
            # Run the heavy GPU work in a thread to not block the event loop
            results = await asyncio.to_thread(
                self.generator.generate_batch,
                texts=texts,
                audio_ref_path=ref_path,
                voice_clone_prompt_bytes=prompt_bytes,
                **gen_kwargs
            )

            duration = round(time.time() - start_time, 2)
            self.total_batches_processed += 1
            self.total_items_processed += batch_size
            print(f"✅ Batch of {batch_size} done in {duration}s "
                  f"(total batches: {self.total_batches_processed}, total items: {self.total_items_processed})")

            # Deliver results to each waiting request
            for job, wav_bytes in zip(batch, results):
                if not job.future.done():
                    job.future.set_result(wav_bytes)

        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            # If batched inference fails, fall back to processing one by one
            await self._fallback_sequential(batch, ref_path, prompt_bytes, gen_kwargs)

    async def _fallback_sequential(self, batch: List[BatchJob], ref_path, prompt_bytes, gen_kwargs):
        """Fallback: process each item individually if batch inference fails."""
        print(f"⚠️ Falling back to sequential processing for {len(batch)} items...")
        for job in batch:
            try:
                wav_bytes = await asyncio.to_thread(
                    self.generator.generate,
                    text=job.text,
                    audio_ref_path=ref_path,
                    voice_clone_prompt_bytes=prompt_bytes,
                    **gen_kwargs
                )
                if not job.future.done():
                    job.future.set_result(wav_bytes)
            except Exception as e2:
                if not job.future.done():
                    job.future.set_exception(e2)
