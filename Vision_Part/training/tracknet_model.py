
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class TrackNet(nn.Module):

    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()

        self.enc1 = nn.Sequential(ConvBlock(in_channels, 64), ConvBlock(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = nn.Sequential(ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)

        self.enc5 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.pool5 = nn.MaxPool2d(2, 2)

        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec6 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))

        self.up7 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec7 = nn.Sequential(ConvBlock(512, 256), ConvBlock(256, 256), ConvBlock(256, 256))

        self.up8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec8 = nn.Sequential(ConvBlock(256, 128), ConvBlock(128, 128))

        self.up9 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec9 = nn.Sequential(ConvBlock(128, 64), ConvBlock(64, 64))

        self.up10 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec10 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]

        pad_h = (32 - orig_h % 32) % 32
        pad_w = (32 - orig_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.enc1(x);  x = self.pool1(x)
        x = self.enc2(x);  x = self.pool2(x)
        x = self.enc3(x);  x = self.pool3(x)
        x = self.enc4(x);  x = self.pool4(x)
        x = self.enc5(x);  x = self.pool5(x)

        x = self.up6(x);   x = self.dec6(x)
        x = self.up7(x);   x = self.dec7(x)
        x = self.up8(x);   x = self.dec8(x)
        x = self.up9(x);   x = self.dec9(x)
        x = self.up10(x);  x = self.dec10(x)

        x = self.final(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]

        return x

if __name__ == "__main__":
    model = TrackNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TrackNet: {total_params:,} total params, {trainable:,} trainable")

    dummy = torch.randn(1, 9, 720, 1280)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 1, 720, 1280), f"Unexpected output shape: {out.shape}"
    print("Forward pass OK [PASS]")

    if torch.cuda.is_available():
        model = model.cuda()
        dummy = torch.randn(2, 9, 720, 1280, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda"):
            out = model(dummy)
            loss = out.mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory (batch=2, AMP): {peak:.2f} GB")
