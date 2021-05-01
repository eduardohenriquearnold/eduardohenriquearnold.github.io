---
title: "Data Study Group Final Report: Greenvest Solutions"
date: 2021-02-11
publishDate: 2021-02-11T09:14:31.801712Z
authors: ["Eduardo Arnold", "Jiaxin Chen", "Ivan Croydon-Veleslavov", "Anurag Deshpande", "Tanaya Guha", "Yixuan He", "Ben Moseley", "Gim Seng Ng", "Luke Prince", "Thomas Statham", "Peter Strong"]
publication_types: ["1"]
abstract: ""
featured: false
publication: "*September 2020 Alan Turing Data Study Group*"
url_pdf : https://www.turing.ac.uk/research/publications/data-study-group-final-report-greenvest-solutions
---

Greenvest is on a mission to accelerate renewable energy adoption worldwide. The start-up provides strategic technology to plan, monitor, and assess clean energy projects globally.

When deciding where to build a new wind farm, a company should assess the wind resources of an area with the minimum uncertainty possible. This is usually done in two steps, firstly, a preliminary assessment is carried out utilizing low-resolution wind maps which are derived from ground stations. Secondly, at least one met(-eorological) mast is installed at the most promising locations to record at least one year of current wind data.

This approach presents several problems. One one hand, even if the wind maps encompass a large temporal frame, they are often inaccurate and poorly interpolated to the desired location. On the other hand, met masts offer precise measurements of wind but they are expensive to be set up and may produce an inaccurate prediction for the long term production of the wind farm due to the year-by-year variability of wind resources.

A newer approach to this problem is to use mesoscale models and computational fluid dynamics to interpolate satellite-derived data but this is a computationally expensive method that often underperforms for complex terrains and coarse grids of satellite data. A promising solution is to use Machine Learning to predict wind resources at a certain location starting from satellite measured data.

Thus, the aim of this challenge is to forecast wind resources close to the ground for a specific location. The key difficulty is to understand how terrain data and surface roughness could be used to train a model that interpolates the satellite data to the desired location.

