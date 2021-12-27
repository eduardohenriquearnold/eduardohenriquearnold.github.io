---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Cooperative Perception for 3D Object Detection in Driving Scenarios using Infrastructure Sensors"
authors: ["Eduardo Arnold", "Mehrdad Dianati", "Robert de Temple", "Saber Fallah"]
date: 2020-10-16
publishDate: 2020-10-16T09:14:31.801192Z


# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

abstract: "3D object detection is a common function within the perception system of an autonomous vehicle and outputs a list of 3D bounding boxes around objects of interest. Various 3D object detection methods have relied on fusion of different sensor modalities to overcome limitations of individual sensors. However, occlusion, limited field-of-view and low-point density of the sensor data cannot be reliably and cost-effectively addressed by multi-modal sensing from a single point of view. Alternatively, cooperative perception incorporates information from spatially diverse sensors distributed around the environment as a way to mitigate these limitations. This article proposes two schemes for cooperative 3D object detection using single modality sensors. The early fusion scheme combines point clouds from multiple spatially diverse sensing points of view before detection. In contrast, the late fusion scheme fuses the independently detected bounding boxes from multiple spatially diverse sensors. We evaluate the performance of both schemes, and their hybrid combination, using a synthetic cooperative dataset created in two complex driving scenarios, a T-junction and a roundabout. The evaluation shows that the early fusion approach outperforms late fusion by a significant margin at the cost of higher communication bandwidth. The results demonstrate that cooperative perception can recall more than 95% of the objects as opposed to 30% for single-point sensing in the most challenging scenario. To provide practical insights into the deployment of such system, we report how the number of sensors and their configuration impact the detection performance of the system."
publication: "*IEEE Transactions on Intelligent Transportation Systems*"
url_code: https://github.com/eduardohenriquearnold/coop-3dod-infra
url_dataset: https://wrap.warwick.ac.uk/159053/
url_pdf: https://ieeexplore.ieee.org/document/9228884
url_code: https://github.com/eduardohenriquearnold/coop-3dod-infra
links:
  - name: Preprint
    url: 'https://arxiv.org/pdf/1912.12147.pdf'
---

