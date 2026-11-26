import os

if not os.environ.get('OMP_NUM_THREADS', '').isdigit() or int(os.environ['OMP_NUM_THREADS']) < 1:
    os.environ['OMP_NUM_THREADS'] = '4'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from args import parser


if __name__ == '__main__':
    args = parser().parse_args()
    from engine import train
    train(args)
