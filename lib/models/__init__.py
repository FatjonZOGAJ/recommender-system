from lib.models import autoencoder, fm, kernel_net, nmf, svd, autorec

models = {'svd': svd,
          'als': nmf,
          'kernel_net': kernel_net,
          'autoencoder': autoencoder,
          'autorec': autorec,
          'fm': fm
          }
