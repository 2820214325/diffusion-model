import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from diff import ContextUnet, ddpm_schedules


# 假设 ContextUnet 和 DDPM 类已经定义在这里，和之前提供的定义相同

def load_model_and_generate_images(model_path, target_class, n_sample=1, device='cuda', guide_w=0.0):
    n_classes = 10  # MNIST数据集中有10个类
    n_feat = 128  # 使用训练时相同的特征维度
    n_T = 400  # 使用训练时相同的扩散步骤数
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        size = (1, 28, 28)  # MNIST图像的尺寸
        x_gen, _ = ddpm.sample(n_sample, size, device, guide_w=guide_w, target_class=target_class)  # 指定生成类别为 3

        grid = make_grid(x_gen * -1 + 1, nrow=10)  # 将生成的数字图片组合成网格
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis('off')
        plt.show()
        save_image(grid, 'generated_images.png')
        print("Saved generated images to 'generated_images.png'")


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )

        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, target_class=3):  # 添加 target_class 参数
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.full((n_sample,), target_class, dtype=torch.long).to(device)  # 全部使用类别 3

        context_mask = torch.zeros_like(c_i).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        x_i_store = []

        for i in tqdm(range(self.n_T, 0, -1), desc="Generating images"):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_class=8 #生成的数字
    load_model_and_generate_images('./data/diffusion_outputs10/model_9.pth', target_class, n_sample=1, device=device,
                                   guide_w=0.0)
