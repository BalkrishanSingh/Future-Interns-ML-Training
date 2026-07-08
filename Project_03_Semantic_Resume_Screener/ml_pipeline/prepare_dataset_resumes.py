"""Prepare demo resume collection from the Kaggle Resume Dataset.

Extracts real PDF resumes focusing on INFORMATION-TECHNOLOGY and Software/CSE
ENGINEERING from the Kaggle Resume Dataset (snehaanbhawal/resume-dataset).
Packages them into demo_resumes/ with manifest.json and demo_resumes.zip.
"""

import os
import glob
import json
import shutil
import zipfile
import pypdf

from pathlib import Path

def prepare_dataset_resumes(limit: int = 150):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolves dataset path via env var, kagglehub cache, or kagglehub API
    dataset_env = os.environ.get('KAGGLE_RESUME_DATASET_DIR')
    if dataset_env and os.path.exists(dataset_env):
        dataset_base = dataset_env
    else:
        cache_default = Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'snehaanbhawal' / 'resume-dataset' / 'versions' / '1' / 'data' / 'data'
        if cache_default.exists():
            dataset_base = str(cache_default)
        else:
            try:
                import kagglehub
                downloaded = kagglehub.dataset_download('snehaanbhawal/resume-dataset')
                dataset_base = os.path.join(downloaded, 'data', 'data')
            except Exception:
                dataset_base = str(cache_default)
    
    it_dir = os.path.join(dataset_base, 'INFORMATION-TECHNOLOGY')
    eng_dir = os.path.join(dataset_base, 'ENGINEERING')

    output_dir = os.path.join(base_dir, 'demo_resumes')
    portal_public_demo = os.path.join(base_dir, 'web_portal', 'public', 'demo_resumes')
    portal_public_zip = os.path.join(base_dir, 'web_portal', 'public', 'demo_resumes.zip')
    root_zip = os.path.join(base_dir, 'demo_resumes.zip')

    # Reset output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(portal_public_demo):
        shutil.rmtree(portal_public_demo)
    os.makedirs(portal_public_demo, exist_ok=True)

    it_files = sorted(glob.glob(os.path.join(it_dir, '*.pdf')))
    print(f'Found {len(it_files)} real Information Technology PDF resumes.')

    eng_files = sorted(glob.glob(os.path.join(eng_dir, '*.pdf')))
    cse_eng_files = []
    for f in eng_files:
        try:
            reader = pypdf.PdfReader(f)
            t_low = ''.join(p.extract_text() for p in reader.pages[:2]).lower()
            if any(k in t_low for k in ['software', 'systems', 'computer', 'network', 'firmware', 'database', 'developer', 'programmer', 'it ', 'web', 'data', 'algorithm']):
                cse_eng_files.append(f)
        except Exception:
            continue
    print(f'Found {len(cse_eng_files)} real Software/CSE Engineering PDF resumes.')

    # Combine: all IT resumes + CSE engineering resumes up to limit
    selected_files = []
    for f in it_files:
        selected_files.append((f, 'INFORMATION-TECHNOLOGY'))
    for f in cse_eng_files:
        if len(selected_files) >= limit:
            break
        selected_files.append((f, 'ENGINEERING'))

    selected_files = selected_files[:limit]
    print(f'Selected {len(selected_files)} real Software/CSE resumes for demo collection.')

    manifest = []
    copied_files = []

    for src_path, category in selected_files:
        raw_filename = os.path.basename(src_path)
        cand_id = os.path.splitext(raw_filename)[0]

        # Extract title from first page
        title = 'Software / IT Specialist'
        try:
            reader = pypdf.PdfReader(src_path)
            pages_count = len(reader.pages)
            first_page_text = reader.pages[0].extract_text()
            lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
            if lines:
                title = lines[0][:80]
        except Exception:
            pages_count = 1

        clean_title = ''.join(c if c.isalnum() else '_' for c in title).strip('_')
        dest_filename = f'{category}_{cand_id}_{clean_title[:30]}.pdf'
        dest_path = os.path.join(output_dir, dest_filename)

        shutil.copyfile(src_path, dest_path)
        copied_files.append((dest_filename, dest_path))

        manifest.append({
            'candidate_id': cand_id,
            'filename': dest_filename,
            'title': title,
            'category': category,
            'file_size_bytes': os.path.getsize(dest_path),
            'pages': pages_count,
            'source_dataset': 'snehaanbhawal/resume-dataset (Kaggle)'
        })

    # Save manifest.json
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Build ZIP archive
    print(f'Packaging {len(copied_files)} real resumes into demo_resumes.zip...')
    with zipfile.ZipFile(root_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname='manifest.json')
        for fname, fpath in copied_files:
            zf.write(fpath, arcname=f'resumes/{fname}')

    shutil.copyfile(root_zip, portal_public_zip)

    # Copy top 25 to web_portal/public/demo_resumes/ for instant portal testing
    top_sample = manifest[:25]
    with open(os.path.join(portal_public_demo, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(top_sample, f, indent=2)

    for item in top_sample:
        src = os.path.join(output_dir, item['filename'])
        dst = os.path.join(portal_public_demo, item['filename'])
        shutil.copyfile(src, dst)

    print(f'Completed preparing {len(manifest)} real dataset resumes.')
    print(f'  - Folder: {output_dir}/ ({len(manifest)} PDFs + manifest.json)')
    print(f'  - Root Zip: {root_zip} ({os.path.getsize(root_zip) / (1024*1024):.2f} MB)')
    print(f'  - Web Public Zip: {portal_public_zip}')
    print(f'  - Web Public Demo Sample: {portal_public_demo}/ (25 PDFs)')

if __name__ == '__main__':
    prepare_dataset_resumes(limit=150)
