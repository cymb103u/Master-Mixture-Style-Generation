from visdom import Visdom
import numpy as np
import torchvision.utils as vutils
import torch
class VisdomPlotter(object):

    """Plots to Visdom"""

    def __init__(self, env_name='gan'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='epoch'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

    def draw(self, var_name, images,display_image_num=8):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images,nrow=display_image_num, padding=0,\
                env=self.env, opts=dict(title=f'{var_name}', caption=f'{var_name}'))
        else:
            self.viz.images(images,nrow=display_image_num,padding=0,\
                 env=self.env, win=self.plots[var_name],opts=dict(title=f'{var_name}', caption=f'{var_name}'))


class Logger(object):
    def __init__(self, vis_screen):
        self.viz = VisdomPlotter(env_name=vis_screen)
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def log_iteration_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss):
        print("Epoch: %d, Gen_iteration: %d, d_loss= %f, g_loss= %f, real_loss= %f, fake_loss = %f" %
              (epoch, gen_iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_loss, fake_loss))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())

    def log_iteration_gan(self, epoch, d_loss, g_loss, real_score, fake_score):
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            fake_score.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(fake_score.data.cpu().mean())

    def plot_epoch(self, epoch):
        self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.hist_D = []
        self.hist_G = []

    def plot_epoch_w_scores(self, epoch):
        self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.viz.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
        self.viz.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def display(self,train_sample,images_interp,display_image_num):
        """
        train_sample : x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2
                        0     1          2       3    4      5         6      7 
        """
        # x_a, x_b,x_ba1(fix), x_ba2(rand)
        a_style_b_content = torch.cat((train_sample[0],train_sample[4],train_sample[6],train_sample[7]),0)
        a_style_b_content = vutils.make_grid(a_style_b_content.data, nrow=display_image_num, padding=0, normalize=True)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        # a_style_b_content = a_style_b_content.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        # x_b, x_a,x_ab1(fix), x_ab2(rand)
        b_style_a_content = torch.cat([train_sample[4],train_sample[0],train_sample[2],train_sample[3]],0)
        b_style_a_content = vutils.make_grid(b_style_a_content.data, nrow=display_image_num, padding=0, normalize=True)
        # b_style_a_content = b_style_a_content.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        self.viz.draw('a_style_b_content', a_style_b_content,display_image_num)
        self.viz.draw('b_style_a_content', b_style_a_content,display_image_num)

        """
        images_interp:x_a[rand_num_a].unsqueeze(0),x_b[rand_num_b].unsqueeze(0),
                        c_a_fix, c_a_real, c_b_fix, c_b_real
        """
        image_a_and_b = torch.cat([images_interp[0],images_interp[1]],0)
        image_a_and_b = vutils.make_grid(image_a_and_b.data, nrow=2, padding=0, normalize=True)
        self.viz.draw('image_a_and_b',image_a_and_b)
        c_a_fix = vutils.make_grid(images_interp[2].data,nrow=6, padding=0, normalize=True)
        c_a_real = vutils.make_grid(images_interp[3], nrow=6, padding=0, normalize=True)
        c_b_fix = vutils.make_grid(images_interp[4], nrow=6, padding=0, normalize=True)
        c_b_real =  vutils.make_grid(images_interp[5], nrow=6, padding=0, normalize=True)
        self.viz.draw('c_a_fix',c_a_fix)
        self.viz.draw('c_a_real',c_a_real)
        self.viz.draw('c_b_fix',c_b_fix)
        self.viz.draw('c_b_real',c_b_real)

if __name__ == "__main__":
    print(__doc__)