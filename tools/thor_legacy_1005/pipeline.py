"""Camera-then-depth scheduling inside the historical whole-frame graph."""
import torch


def forward_pipelined(
    model, static_input, *, side_stream, num_frame_for_scale,
    num_frame_per_block, causal_inference, use_nvtx=False,
    probe_head_mode="serial_camera_then_depth",
):
    if probe_head_mode != "serial_camera_then_depth":
        raise ValueError("Thor capture requires both heads in camera-then-depth order")
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("Thor capture is inference-only")
    images = static_input.unsqueeze(0) if static_input.ndim == 4 else static_input
    main_stream = torch.cuda.current_stream()
    if use_nvtx:
        torch.cuda.nvtx.range_push("aggregator")
    aggregated_tokens_list, patch_start_idx = model._aggregate_features(
        images, num_frame_for_scale=num_frame_for_scale,
        num_frame_per_block=num_frame_per_block,
    )
    if use_nvtx:
        torch.cuda.nvtx.range_pop()
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream):
        if use_nvtx:
            torch.cuda.nvtx.range_push("camera_head")
        cam_pred = model._predict_camera(
            aggregated_tokens_list, mask=None, causal_inference=causal_inference,
            num_frame_for_scale=num_frame_for_scale, sliding_window_size=None,
            num_frame_per_block=num_frame_per_block, gather_outputs=True,
        )
        if use_nvtx:
            torch.cuda.nvtx.range_pop()
    main_stream.wait_stream(side_stream)
    if use_nvtx:
        torch.cuda.nvtx.range_push("depth_head")
    depth_pred = model._predict_depth(
        aggregated_tokens_list, images, patch_start_idx, gather_outputs=True,
    )
    if use_nvtx:
        torch.cuda.nvtx.range_pop()
    return {**cam_pred, **depth_pred, "images": images}
