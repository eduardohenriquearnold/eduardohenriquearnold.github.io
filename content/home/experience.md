+++
# Experience widget.
widget = "experience"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true # Activate this widget? true/false
weight = 25  # Order that this section will appear.

title = "Experience"
subtitle = ""

# Date format for experience
#   Refer to https://sourcethemes.com/academic/docs/customization/#date-format
date_format = "Jan 2006"

# Experiences.
#   Add/remove as many `[[experience]]` blocks below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin/end multi-line descriptions with 3 quotes `"""`.
[[experience]]
  title = "Research Assistant"
  company = "University of Warwick"
  company_url = "https://warwick.ac.uk/"
  location = "Coventry, United Kingdom"
  date_start = "2018-08-01"
  date_end = "2021-06-01"
  description = """
  The L3 Pilot European Consortium studies the impact of vehicle automation with key European OEMs. My main responsibilities were:
  * Lead the creation of a responsive online platform to support data visualisation of piloting data across the project.
  * Attend international meetings with key partners in the automotive sector to meet project requirements and report results.
 """

[[experience]]
  title = "Teaching Assistant"
  company = "University of Warwick"
  company_url = "https://warwick.ac.uk/"
  location = "Coventry, United Kingdom"
  date_start = "2019-05-01"
  date_end = "2020-04-01"
  description = """
  * Prepared and delivered tutorials on PyTorch and fundamentals of machine learning;
  * Prepared module assessments and marked students; 
  * Created a Dockerized Jupyterhub platform for students to run their code remotely.
  """

[[experience]]
  title = "PhD Research Intern"
  company = "Niantic"
  company_url = "https://nianticlabs.com/"
  location = "London, UK"
  date_start = "2021-06-01"
  date_end = "2022-01-01"
  description = """
  Used computer vision and machine learning techniques for augmented reality applications; 
  * Created a new benchmark and dataset for visual re-localization with a team of skilled researchers;
  * Trained and evaluated several families of visual localisation methods on this benchmark;
  * Used cloud infrastructure to train and evaluate deep learning models;
  * Presented research outcomes to a cross-disciplinary audience;
  * Paper published at ECCV 2022
  """

[[experience]]
  title = "Machine Learning Engineer"
  company = "Niantic"
  company_url = "https://nianticlabs.com/"
  location = "London, UK"
  date_start = "2022-05-01"
  date_end = "2023-12-01"
  description = """
  * Ran several state-of-the-art SLAM methods on internal datasets and devised an evaluation protocol to compare them.
  * Designed and developed a localization pipeline to align scans to a large scale reference reconstruction in challenging environments.
  * Created a renderer that process >100bi points in less than 20s.
  * Helped adding Lidar-based terms to COLMAP, reducing re-projection median errors from 60px to 3px.
  * Trained and evaluated different NERF variants using aerial views.
  * Performed Camera-IMU calibration which helped to identify time-sync issues.
  * Created multiple Argo/K8s workflows to process data at scale using cloud infrastructure (GCP).
  """

[[experience]]
  title = "Lead Computer Vision Engineer"
  company = "Cartesian"
  company_url = "https://www.cartesian.systems/"
  location = "Cambridge, USA"
  date_start = "2024-02-01"
  date_end = ""
  description = """
  * Optimized a Structure-from-Motion pipeline resulting in a reduction of mapping times from 655min to 102min, a 6-fold speed up.
  * Created benchmark tool that creates ground-truth maps and quantify the quality of mapping and localization with high accuracy.
  * Wrote new production visual re-localization pipeline with PyTorch resulting in 3x faster, 10x higher throughput, using 100x less memory than previous baseline and saving more than 95% in cloud compute costs.
  * Designed and implemented an end-to-end mapping pipeline that process raw data into production-ready SfM maps using Ray workflows on Azure Kubernetes Service (AKS).
  * Trained a global feature model (NetVLAD) for image retrieval in specific data domains, resulting in 10% improvement in localization performance.
  * Designed and implemented a system architecture using REDIS, Helm and K8s to allow scaling the re-localization service to a very large number of simultaneous requests across thousands of maps, whilst maintaining QoS.
  """

+++
