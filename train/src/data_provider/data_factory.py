from src.data_provider.data_loader import *


data_dict = {
    'Benchmark': Dataset_Benchmark,
    'Benchmark_Cond': Dataset_Benchmark_Conditional,
    'Benchmark_FinDiff': Dataset_Benchmark_FinDiff,
    'Benchmark_LTR': Dataset_Benchmark_LTR,
    'RegimeWindows': Dataset_RegimeWindows,
}


def data_provider(args, col=None):
    Data = data_dict[args.data]
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size  # bsz for train

    data_set = Data(
        data_path=args.data_path,
        regime_path=args.regime_path,
        seq_len=args.seq_len,
        scale=args.scale,
        sample_multiplier=args.sample_multiplier,
        start_date=args.start_date,
        end_date=args.end_date,
        cond_type=args.cond_type,
        n_regimes=args.n_regimes,
        prev_len=args.prev_len,
        col=col,  # select an asset
        seq_len_multiplier=args.seq_len_multiplier,
    )
    print(f'DataLoader: {len(data_set)}')
    if args.data == 'RegimeWindows' and len(data_set) < batch_size:
        drop_last = False
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=drop_last,
        shuffle=shuffle_flag,
    )

    return data_set, data_loader


def data_provider_Joint_KL_Div(X, Y, num_samples=10000, batch_size=1024):
    data_set = Dataset_Joint_KL_Div(X, Y, num_samples=num_samples)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        shuffle=True,
    )

    return data_set, data_loader