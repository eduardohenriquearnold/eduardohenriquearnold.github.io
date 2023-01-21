---
title: "Map-free Visual Relocalization: Metric Pose Relative to a Single Image"
date: 2022-10-23
publishDate: 2022-10-23T13:14:31.801492Z
authors: ["Eduardo Arnold", "Jamie Wynn", "Sara Vicente", "Guillermo Garcia-Hernando", "Áron Monszpart", "Victor Prisacariu", "Daniyar Turmukhambetov", "Eric Brachmann"]
publication_types: ["1"]
abstract: "Can we relocalize in a scene represented by a single reference image? Standard visual relocalization requires hundreds of images and scale calibration to build a scene-specific 3D map. In contrast, we propose Map-free Relocalization, i.e., using only one photo of a scene to enable instant, metric scaled relocalization. Existing datasets are not suitable to benchmark map-free relocalization, due to their focus on large scenes or their limited variability. Thus, we have constructed a new dataset of 655 small places of interest, such as sculptures, murals and fountains, collected worldwide. Each place comes with a reference image to serve as a relocalization anchor, and dozens of query images with known, metric camera poses. The dataset features changing conditions, stark viewpoint changes, high variability across places, and queries with low to no visual overlap with the reference image. We identify two viable families of existing methods to provide baseline results: relative pose regression, and feature matching combined with single-image depth prediction. While these methods show reasonable performance on some favorable scenes in our dataset, map-free relocalization proves to be a challenge that requires new, innovative solutions."
featured: false
publication: "ECCV 2022 - European Conference on Computer Vision"
url_code: https://github.com/nianticlabs/map-free-reloc
url_dataset: https://research.nianticlabs.com/mapfree-reloc-benchmark
url_pdf: https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/MapFreeReloc-ECCV22-paper.pdf
links:
  - name: Preprint
    url: 'https://arxiv.org/abs/2210.05494'
  - name: Video
    url: 'https://storage.cloud.google.com/niantic-lon-static/research/map-free-reloc/MapFreeReloc-ECCV22-video.mp4'
---
---