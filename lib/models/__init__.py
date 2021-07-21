from lib.models import svd, nmf, kernel_net, autoencoder, fm, fm_rel

models = {'svd': svd,
          'als': nmf,
          'kernel_net': kernel_net,
          'autoencoder': autoencoder,
          'fm': fm,
          'fm_rel': fm_rel
          }
