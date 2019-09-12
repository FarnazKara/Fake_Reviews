# Fake_Reviews
For training the model, run ```python train.py --config config/nmt.ini```


For prediction, run translate.py with --config /config/vnmt.ini


This code is based on https://github.com/wang-h/Variational-NMT

References
Su, Jinsong, et al. "Variational Recurrent Neural Machine Translation." arXiv preprint arXiv:1801.05119 (2018).

Zhang, Biao, et al. "Variational neural machine translation." arXiv preprint arXiv:1605.07869 (2016)

Differences
For Variational NMT, I did not use the mean-pooling for both sides (source and target). I tested only using the last source hidden state is sufficient to achieve good performance.

For Variational Recurrent NMT, I tested only using the current RNN state is sufficient to achieve good performance.

The paper

Yang, Zichao, et al. "Improved variational autoencoders for text modeling using dilated convolutions." arXiv preprint arXiv:1702.08139 (2017).

explains the reason why use GRU instead of LSTM for building RNN cell, in general, VAE-LSTM-decoder performs worse than vanilla-LSTM-decoder.
