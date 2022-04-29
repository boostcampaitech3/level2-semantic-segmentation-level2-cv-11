import segmentation_models_pytorch as smp


def build_model(args):
    decoder = getattr(smp, args.decoder)
    model = decoder(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=11,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    return model, preprocessing_fn

