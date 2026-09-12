import torch


def apply_feature_overrides(opt, args):
    """Allow feature-set experiments without editing config.yml."""
    for name in ["a_feat_dir", "t_feat_dir", "a_feat_dim", "t_feat_dim", "ctx_mode", "clip_length", "max_a_l"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(opt, name, value)
    return opt


def feature_config_summary(opt):
    return {
        "a_feat_dir": opt.a_feat_dir,
        "t_feat_dir": opt.t_feat_dir,
        "a_feat_dim": opt.a_feat_dim,
        "t_feat_dim": opt.t_feat_dim,
        "ctx_mode": opt.ctx_mode,
        "clip_length": opt.clip_length,
        "max_a_l": opt.max_a_l,
    }


def make_dataset_config(opt, data_path):
    return dict(
        data_path=data_path,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )


def unpack_batch(batch, device, a_feat_dim):
    batch_meta, batched_inputs = batch

    query_feat, query_mask = batched_inputs["query_feat"]
    audio_feat, audio_mask = batched_inputs["audio_feat"]

    query_feat = query_feat.to(device)
    query_mask = query_mask.to(device)
    audio_feat = audio_feat.to(device)
    audio_mask = audio_mask.to(device)

    audio_feat_clap = audio_feat[:, :, :a_feat_dim]
    query_valid_mask = query_mask.long()
    audio_valid_mask = audio_mask.long()

    return batch_meta, query_feat, query_valid_mask, audio_feat, audio_feat_clap, audio_valid_mask


def torch_load_trusted(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
