from lib.models import autoencoder, fm, kernel_net, nmf, svd

models = {'svd': svd,
          'als': nmf,
          'kernel_net': kernel_net,
          'autoencoder': autoencoder,
          'fm': fm
          }
