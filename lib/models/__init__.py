from lib.models import autoencoder, fm, kernel_net, nmf, svd, autorec, ncf

models = {'svd': svd,
          'nmf': nmf,
          'kernel_net': kernel_net,
          'autoencoder': autoencoder,
          'autorec': autorec,
          'fm': fm,
          'ncf': ncf
          }
