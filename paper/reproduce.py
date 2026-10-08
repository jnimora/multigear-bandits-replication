"""Rebuild the current manuscript figures/tables and verify compact evidence."""
from pathlib import Path
import argparse,importlib.util,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT.parent/'results/local/paper')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    work=out/'analysis';shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'tables').mkdir();(out/'figures').mkdir()
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib-cache'))
    subprocess.run([sys.executable,str(work/'verify_allocation.py')],check=True)
    subprocess.run([sys.executable,str(work/'verify_block.py')],check=True)
    subprocess.run([sys.executable,str(work/'blocked_multigear_evidence/independent_review.py')],check=True)
    sys.path.insert(0,str(work))
    import report_blocked_multigear as multi
    multi.build(work/'blocked_multigear_evidence',out,work/'blocked_binary_evidence')
    import report_blocked_binary as binary
    binary.validate();binary.tables();binary.render()
    subprocess.run([sys.executable,str(work/'plot_applications.py')],check=True)
    print('Verified evidence and regenerated figures/tables:',out)
if __name__=='__main__':main()
