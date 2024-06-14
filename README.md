
# AlphaZeroChomp


This project is a reproduction of the paper AlphaZero (https://arxiv.org/pdf/1712.01815.pdf) for the game of Chomp (https://en.wikipedia.org/wiki/Chomp ). It uses a graph Markov Decision Process with a directed graph and not a normal graph to improve compuational efficiency. It implements many features like Dynamic Batching. In this version I did not support for GPU training but I plan to do so. 


## Alessandro Canzonieri

- [@babayaga102](https://github.com/babayaga102)


## Deployment

To deploy this project run
Move to the ./src

```
  pip install -r requirements.txt
```
To train and play, run:
```
  python main.py --mode train_and_play
```

To only play, run:
```
  python main.py --mode play
```




