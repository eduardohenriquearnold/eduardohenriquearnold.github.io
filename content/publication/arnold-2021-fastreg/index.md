---
title: "Fast and Robust Registration of Partially Overlapping Point Clouds"
date: 2021-12-23
publishDate: 2021-12-23T13:14:31.801492Z
authors: ["Eduardo Arnold", "Sajjad Mozaffari", "Mehrdad Dianati"]
publication_types: ["1"]
abstract: "Real-time registration of partially overlapping point clouds has emerging applications in cooperative perception for autonomous vehicles and multi-agent SLAM. The relative translation between point clouds in these applications is higher than in traditional SLAM and odometry applications, which challenges the identification of correspondences and a successful registration. In this paper, we propose a novel registration method for partially overlapping point clouds where correspondences are learned using an efficient point-wise feature encoder, and refined using a graph-based attention network. This attention network exploits geometrical relationships between key points to improve the matching in point clouds with low overlap. At inference time, the relative pose transformation is obtained by robustly fitting the correspondences through sample consensus. The evaluation is performed on the KITTI dataset and a novel synthetic dataset including low-overlapping point clouds with displacements of up to 30m. The proposed method achieves on-par performance with state-of-the-art methods on the KITTI dataset, and outperforms existing methods for low overlapping point clouds. Additionally, the proposed method achieves significantly faster inference times, as low as 410ms, between 5 and 35 times faster than competing methods. Our code and dataset will be available at https://github.com/eduardohenriquearnold/fastreg."
featured: false
publication: "*2021 IEEE Robotics and Automation Letters (RA-L)*"
url_code: https://github.com/eduardohenriquearnold/fastreg
url_dataset: https://github.com/eduardohenriquearnold/CODD
url_pdf: https://ieeexplore.ieee.org/document/9662220
links:
  - name: Preprint
    url: 'https://arxiv.org/abs/2112.09922'
---
---

