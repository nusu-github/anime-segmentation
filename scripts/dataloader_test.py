import argparse
import time

from anime_segmentation.data_module import AnimeSegDataModule


def benchmark_dataloader(
    data_dir: str = "dataset/",
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 640,
    num_batches: int = 100,
    warmup_batches: int = 10,
    *,
    with_trimap: bool = True,
) -> None:
    """Benchmark the dataloader throughput."""
    data_module = AnimeSegDataModule(
        data_dir=data_dir,
        fg_dir="fg",
        bg_dir="bg",
        img_dir="imgs",
        mask_dir="masks",
        fg_ext=".png",
        bg_ext=".jpg",
        img_ext=".jpg",
        mask_ext=".jpg",
        data_split=0.95,
        img_size=img_size,
        batch_size_train=batch_size,
        num_workers_train=num_workers,
        with_trimap=with_trimap,
    )
    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()

    print(f"Dataset size: {len(train_dataloader.dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Image size: {img_size}")
    print(f"With trimap: {with_trimap}")
    print("-" * 50)

    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    dataloader_iter = iter(train_dataloader)
    for _ in range(warmup_batches):
        try:
            next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            next(dataloader_iter)

    # Benchmark
    print(f"Benchmarking ({num_batches} batches)...")
    batch_times = []
    total_samples = 0

    start_time = time.perf_counter()
    for i in range(num_batches):
        batch_start = time.perf_counter()
        try:
            data = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            data = next(dataloader_iter)

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
        total_samples += data["image"].shape[0]

        if (i + 1) % 20 == 0:
            print(f"  Batch {i + 1}/{num_batches}: {batch_time * 1000:.1f}ms")

    total_time = time.perf_counter() - start_time

    # Results
    print("-" * 50)
    print("Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total samples: {total_samples}")
    print(f"  Throughput: {total_samples / total_time:.1f} samples/sec")
    print(f"  Avg batch time: {sum(batch_times) / len(batch_times) * 1000:.1f}ms")
    print(f"  Min batch time: {min(batch_times) * 1000:.1f}ms")
    print(f"  Max batch time: {max(batch_times) * 1000:.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark dataloader")
    parser.add_argument("--data-dir", type=str, default="dataset/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--with-trimap", action="store_true")
    args = parser.parse_args()

    benchmark_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        with_trimap=args.with_trimap,
    )
