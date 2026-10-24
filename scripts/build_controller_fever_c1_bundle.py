"""Package sealed Top5 and pinned Qwen-only code for Kaggle/Colab."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.controller.fever_c1 import verify_collection

BUNDLE_NAME='controller_fever_c1_gpu_v1'


def cell(kind,text):
    value={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value


def notebook(archive_sha,index_sha,runtime=None):
    loader=f'''import os, sys, json, hashlib, tarfile, shutil, subprocess
from pathlib import Path
import torch
assert sys.version_info >= (3, 11)
assert torch.cuda.is_available(), '请选择 CUDA GPU，并启用 Internet'
print({{'torch':torch.__version__,'cuda':torch.version.cuda,
       'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}})
WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
NAME={BUNDLE_NAME!r}
EXPECTED_ARCHIVE={archive_sha!r}
EXPECTED_INDEX={index_sha!r}
BUNDLE=WORK/('input_'+EXPECTED_INDEX[:12])/NAME
OUTPUT=WORK/'controller_fever_c1_v1_outputs'
assert not OUTPUT.exists(), '保留旧运行结果；不要在原目录重复推理'
if not BUNDLE.exists():
    archives=list(Path('/kaggle/input').rglob(NAME+'.tar.gz')) if Path('/kaggle/input').exists() else []
    indexes=[p for p in Path('/kaggle/input').rglob('bundle_checksums.json') if p.parent.name==NAME] if Path('/kaggle/input').exists() else []
    if archives:
        assert len(archives)==1
        archive=archives[0]
    elif indexes:
        assert len(indexes)==1
        assert hashlib.sha256(indexes[0].read_bytes()).hexdigest()==EXPECTED_INDEX
        BUNDLE.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(indexes[0].parent,BUNDLE)
        archive=None
    else:
        from google.colab import files
        upload=files.upload()
        archive=WORK/(NAME+'.tar.gz')
        archive.write_bytes(upload[NAME+'.tar.gz'])
    if archive is not None:
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED_ARCHIVE
        BUNDLE.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(BUNDLE.parent,filter='data')
index=BUNDLE/'bundle_checksums.json'
assert hashlib.sha256(index.read_bytes()).hexdigest()==EXPECTED_INDEX
checksums=json.loads(index.read_text())
for relative,digest in checksums.items():
    path=Path(relative)
    assert not path.is_absolute() and '..' not in path.parts
    assert hashlib.sha256((BUNDLE/path).read_bytes()).hexdigest()==digest, relative
print('输入全部校验通过:',len(checksums),'个文件；FEVER 200/200/500')
'''
    install='''subprocess.run([sys.executable,'-m','pip','install','-r',str(BUNDLE/'requirements-answer-repair-v8.txt')],check=True)
subprocess.run([sys.executable,'scripts/run_controller_fever_c1.py','--collection','collection',
                '--output-dir',str(OUTPUT),'--preflight'],cwd=BUNDLE,check=True)
'''
    run='''# 先进行虚构输入工程检查，再统一采集 900 条 C1。
assert not OUTPUT.exists()
log=WORK/'controller_fever_c1_v1.log'
command=[sys.executable,'-u','scripts/run_controller_fever_c1.py','--collection','collection',
         '--output-dir',str(OUTPUT)]
with log.open('x',encoding='utf-8') as handle:
    process=subprocess.Popen(command,cwd=BUNDLE,env={**os.environ,'PYTHONUNBUFFERED':'1'},
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:
            print(line,end='',flush=True);handle.write(line);handle.flush()
        returncode=process.wait()
    except BaseException:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
OUTPUT.mkdir(parents=True,exist_ok=True)
shutil.copyfile(log,OUTPUT/'run.log')
shutil.copyfile(BUNDLE/'collection/manifest.json',OUTPUT/'collection_manifest.json')
if returncode:
    if not (OUTPUT/'manifest.json').exists():
        (OUTPUT/'manifest.json').write_text(json.dumps({'state':'infrastructure_failed','returncode':returncode}))
    raise RuntimeError('本次采集失败，保留日志和部分输出；不要挑题重跑')
print('C1 采集完成；此处不计算正确率，也不查看最终集逐题答案。')
'''
    export='''# 失败后也可执行此单元格导出诊断文件。
assert OUTPUT.exists()
result_hashes={str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in OUTPUT.rglob('*') if p.is_file() and p.name!='output_checksums.json'}
(OUTPUT/'output_checksums.json').write_text(json.dumps(result_hashes,indent=2))
archive=Path(shutil.make_archive(str(WORK/'controller_fever_c1_v1_outputs'),'zip',OUTPUT))
print('结果文件:',archive,'SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())
if not Path('/kaggle/working').exists():
    from google.colab import files
    files.download(str(archive))
'''
    intro='''# Controller：FEVER 冻结 900 条 C1 采集

启用 CUDA GPU 与 Internet，添加配套的私有实验包，依次运行。
仅生成已有 Top-5 证据条件下的 C1，不运行增强或评测。开发/选型/最终候选为 200/200/500，输出分开。
代码与输入逐文件校验；模型固定 Qwen2.5-7B-Instruct、NF4、greedy、64 输出 token。
包内无 gold、人工标签、原始大索引或模型权重。保留所有未知回答；失败保留日志和部分输出。
首次输出前代码与输入已冻结。最终候选答案封存，不能据此继续改 prompt 或选策略。
'''
    if runtime:
        intro+=f"\n本版统一使用原生 chunked prefill={runtime['prefill_chunk_size']}，完整保留输入。正式 900 条前额外运行至少 {runtime['long_fixture_min_input_tokens']} 输入 token 的虚构长上下文工程检查。旧失败结果保留，新结果不与旧配置混合。\n"
    values=[cell('markdown',intro)]+[cell('code',x) for x in (loader,install,run,export)]
    for value in values:
        if value['cell_type']=='code':compile(''.join(value['source']),'<notebook-cell>','exec')
    return {'cells':values,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'accelerator':'GPU','colab':{'name':'controller_fever_c1_v1.ipynb'}},'nbformat':4,'nbformat_minor':5}


def build(collection,archive_path,notebook_path,*,root=ROOT):
    collection,archive_path,notebook_path,root=map(Path,(collection,archive_path,notebook_path,root))
    manifest,_=verify_collection(collection,root=root)
    if archive_path.exists() or notebook_path.exists():
        raise FileExistsError('Preserve existing GPU artifacts')
    payload={relative:(root/relative).read_bytes() for relative in manifest['source_files']}
    payload['collection/manifest.json']=(collection/'manifest.json').read_bytes()
    for info in manifest['files'].values():
        payload['collection/'+info['path']]=(collection/info['path']).read_bytes()
    for name in payload:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe bundle path')
    checksums={name:hashlib.sha256(content).hexdigest() for name,content in sorted(payload.items())}
    index=(json.dumps(checksums,sort_keys=True,indent=2)+'\n').encode()
    payload['bundle_checksums.json']=index
    archive_path.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive_path,'x:gz') as handle:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(BUNDLE_NAME+'/'+name);info.size=len(content);info.mode=0o644
            handle.addfile(info,io.BytesIO(content))
    digest=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    index_sha=hashlib.sha256(index).hexdigest()
    notebook_path.parent.mkdir(parents=True,exist_ok=True)
    notebook_path.write_text(json.dumps(notebook(digest,index_sha,manifest.get('generation_runtime')),ensure_ascii=False,indent=2)+'\n')
    return {'archive_sha256':digest,'archive_bytes':archive_path.stat().st_size,
            'checksum_index_sha256':index_sha,'files':len(checksums),'gold_included':False,'n_tasks':900}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(build(args.collection,args.archive,args.notebook),indent=2))


if __name__=='__main__':
    main()
