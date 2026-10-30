"""Read-only capture probe, using the production identity and backend checks."""
import argparse
import json
from pathlib import Path
from capture_windows import enable_dpi_awareness,list_game_windows
from capture_session import snapshot
from dialogue_store import write_png

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['mss','wgc','printwindow'],default='wgc')
    parser.add_argument('--output',default=str(Path(__file__).parent/'validation/game-initial.png'))
    args=parser.parse_args()
    enable_dpi_awareness()
    window=next((w for w in list_game_windows() if not w.is_launcher),None)
    if window is None:
        raise RuntimeError('没有找到已核实身份的异环游戏窗口')
    frame=snapshot(window,backend=args.backend)
    write_png(Path(args.output),frame)
    print(json.dumps(dict(window=vars(window),backend=args.backend,image=args.output,
                         size=[frame.shape[1],frame.shape[0]]),ensure_ascii=True))

if __name__=='__main__':
    main()
