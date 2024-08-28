from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()

wildcard_constraints:
    regionid="\d+"

SAMPLES = list(glob_wildcards("input/{itemid}.jpg").itemid)

# there is currently a bug in snakemake which prevents batching of jobs
# downstream of a checkpoint: https://github.com/snakemake/snakemake/issues/1984
#
# def agg_masks_input(wildcards):
#     masks_directory = checkpoints.create_masks.get(**wildcards).output.masks_directory

#     vals = expand("output/content_crop/{itemid}/{regionid}.png",
#                 zip,
#                 itemid=glob_wildcards(os.path.join(masks_directory, "{itemid}/{regionid}.png")).itemid,
#                 regionid=glob_wildcards(os.path.join(masks_directory, "{itemid}/{regionid}.png")).regionid)
#     return vals

# rule agg_masks:
#     input:
#         agg_masks_input
#     output:
#         masklist="output/masklists/{itemid}.masklist"
#     run:
#         for i in input:
#             shell("printf '{i}\\n' >> {output}")

# this functionality is now in amg.py
#
# rule apply_mask:
#     input:
#         target="input/{itemid}.jpg",
#         mask="output/masks/{itemid}/{regionid}.png",
#     output:
#         path="output/content_crop/{itemid}/{regionid}.png",
#     shell:
#         "python maoii-backend/scripts/mask_apply.py -m {input.mask} -t {input.target} -a -o {output.path}"

rule all:
    input:
        "output/metadata/all_metadata.json",
        "output/embeddings/embeddings.json",
        "output/merged/merged.jsonl",
        #metadata_paths=expand("output/rle_masks/{itemid}.json", itemid=SAMPLES)


rule get_sam_model:
    input:
        HTTP.remote("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", keep_local=True)
    output:
        "models/sam_vit_h_4b8939.pth"
    run:
        shell("mv {input} {output}")

rule get_sam2_model:
    input:
        HTTP.remote("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", keep_local=True)
    output:
        "models/sam2_hiera_base_plus.pt"
    run:
        shell("mv {input} {output}")

rule get_sam2_config:
    input:
        HTTP.remote("https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_b%2B.yaml", keep_local=True)
    output:
        sam2_config="sam2_configs/sam2_hiera_b+.yaml",
        init="sam2_configs/__init__.py"
    run:
        shell("mv {input} {output.sam2_config}")
        shell("touch {output.init}")

# this is the easiest way to dynamically select the correct "create_masks rule"
# other ways involve renaming files or using checkpoints
if config.get("sam_type", None) == "sam":
    ruleorder: create_masks_sam > create_masks_sam2 > create_masks_fastsam
elif config.get("sam_type", None) == "sam2":
    ruleorder: create_masks_sam2 > create_masks_sam > create_masks_fastsam
else:
    ruleorder: create_masks_fastsam > create_masks_sam2 > create_masks_sam

rule create_masks_sam:
    # set --resources gpu_workers=1 in CLI invocation of runner to 
    # prevent out-of-memory on GPU (forces serial execution of this step)    
    input:
        input_filenames=expand("input/{itemid}.jpg", itemid=SAMPLES),
        model_path="models/sam_vit_h_4b8939.pth"
    resources:
        gpu_workers=1
    output:
        masks_directory=directory("output/masks/"),
        content_directory=directory("output/content_crop/"),
        metadata_paths=expand("output/masks/{itemid}/metadata.csv", itemid=SAMPLES)
    shell:
        "python maoii-backend/scripts/amg.py --input-filenames {input.input_filenames} "
        "--output {output.masks_directory} --content-output {output.content_directory} "
        "--sam-type sam --checkpoint {input.model_path} --model-type vit_h"

rule create_masks_sam2:
    # set --resources gpu_workers=1 in CLI invocation of runner to 
    # prevent out-of-memory on GPU (forces serial execution of this step)    
    input:
        input_filenames=expand("input/{itemid}.jpg", itemid=SAMPLES),
        model_path="models/sam2_hiera_base_plus.pt",
        model_config="sam2_configs/sam2_hiera_b+.yaml",
        init="sam2_configs/__init__.py",
    resources:
        gpu_workers=1
    output:
        masks_directory=directory("output/masks/"),
        content_directory=directory("output/content_crop/"),
        metadata_paths=expand("output/masks/{itemid}/metadata.csv", itemid=SAMPLES)
    shell:
        "python maoii-backend/scripts/amg.py --input-filenames {input.input_filenames} "
        "--output {output.masks_directory} --content-output {output.content_directory} "
        "--sam-type sam2 --checkpoint {input.model_path} --sam2-config {input.model_config}"

rule create_masks_fastsam:
    # set --resources gpu_workers=1 in CLI invocation of runner to 
    # prevent out-of-memory on GPU (forces serial execution of this step)    
    input:
        input_filenames=expand("input/{itemid}.jpg", itemid=SAMPLES),
    resources:
        gpu_workers=1
    output:
        masks_directory=directory("output/masks/"),
        content_directory=directory("output/content_crop/"),
        metadata_paths=expand("output/masks/{itemid}/metadata.csv", itemid=SAMPLES)
    shell:
        "python maoii-backend/scripts/amg.py --input-filenames {input.input_filenames} "
        "--output {output.masks_directory} --content-output {output.content_directory} "
        "--sam-type fastsam --checkpoint dummy "

# rule create_rle_masks:
#     # set --resources gpu_workers=1 in CLI invocation of runner to 
#     # prevent out-of-memory on GPU (forces serial execution of this step)    
#     input:
#         input_filenames=expand("input/{itemid}.jpg", itemid=SAMPLES),
#         model_path="models/sam_vit_h_4b8939.pth"
#     resources:
#         gpu_workers=1
#     output:
#         metadata_paths=expand("output/rle_masks/{itemid}.json", itemid=SAMPLES)
#     shell:
#         "python maoii-backend/scripts/amg.py --input-filenames {input.input_filenames} "
#         "--convert-to-rle --output output/rle_masks "
#         "--checkpoint {input.model_path} --model-type vit_h"

rule agg_masklists:
    input:
        "output/metadata/all_metadata.json"
    output:
        "output/masklists/all.masklist"
    shell:
        "find output/content_crop -name '*.png' -type f > {output}"

rule json_metadata:
    input:
        csv_path="output/masks/{itemid}/metadata.csv"
    output:
        json_path="output/metadata/{itemid}.json"
    shell:
        "jq -cR 'split(\",\")' {input.csv_path} "
        "| jq -csf maoii-backend/scripts/j2c.jq "
        "| jq -c '.[].filename = \"{wildcards.itemid}\"' "
        "> {output.json_path}"

rule jsonl_embeddings:
    input:
        "output/embeddings/embeddings.csv"
    output:
        "output/embeddings/embeddings.jsonl"
    shell:
        "jq -cR 'split(\",\")' {input} "
        "| jq -csf maoii-backend/scripts/j2c.jq "
        "| jq -c '.[] | {{\"filenames\":.filenames, \"embeddings\":([. | to_entries[] | select(.key | contains(\"embedding\")) | .value])}}' "
        "> {output}"

rule jsonl_all_metadata:
    input:
        expand("output/metadata/{itemid}.json", itemid=SAMPLES)
    output:
        "output/metadata/all_metadata.jsonl"
    shell:
        "cat {input} "
        "| jq -c .[] "
        "> {output}"

rule json_all_metadata:
    input:
        "output/metadata/all_metadata.jsonl"
    output:
        "output/metadata/all_metadata.json"
    shell:
        "jq -cs '.' {input} "
        "> {output}"

rule json_embeddings:
    input:
        "output/embeddings/embeddings.jsonl"
    output:
        "output/embeddings/embeddings.json"
    shell:
        "jq -cs '.' {input} "
        "> {output}"

rule json_merged:
    input:
        embeddings="output/embeddings/embeddings.json",
        all_metadata="output/metadata/all_metadata.json"
    output:
        "output/merged/merged.json"
    shell:
        "python maoii-backend/scripts/merge_json.py "
        "--embeddings-path {input.embeddings} "
        "--metadata-path {input.all_metadata} "
        "--merged-path {output} "

rule jsonl_merged:
    input:
        "output/merged/merged.json"
    output:
        "output/merged/merged.jsonl"
    shell:
        "jq -c .[] {input} > {output}"

rule train_embeddings:
    resources:
        gpu_workers=1
    threads:
        32
    input:
        "output/masklists/all.masklist",
    output:
        checkpoint="output/checkpoints/latest.ckpt",
    params:
        max_epochs=10,
        precision=16,
        batch_size=256,
    shell:
        "python maoii-backend/scripts/simmim.py "
        "--max-epochs {params.max_epochs} --precision {params.precision} "
        "--batch-size {params.batch_size} --num-workers {threads} "
        "--input-folder output/content_crop --log-folder output/logs; "
        "find output/logs -name last.ckpt -printf \"%T@ %p\n\" | sort -n | "
        "head -n 1 | cut -d ' ' -f2 | xargs -I {{}} cp {{}} {output.checkpoint}"

rule infer_embeddings:
    resources:
        gpu_workers=1
    threads:
        32
    input:
        checkpoint="output/checkpoints/latest.ckpt",
        masklist="output/masklists/all.masklist"
    output:
        embeddings_path="output/embeddings/embeddings.csv"
    params:
        pca_n_dims=128,
        batch_size=128,
    shell:
        "python maoii-backend/scripts/simmim_infer.py "
        "--input-folder output/content_crop --checkpoint {input.checkpoint} "
        "--batch-size {params.batch_size} --num-workers {threads} "
        "--pca-n-dims {params.pca_n_dims} --output-embeddings-file {output.embeddings_path}"
