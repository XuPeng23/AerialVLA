# AerialVLA
Official repository for AerialVLA: A Vision-Language-Action Model for UAV Navigation via Minimalist End-to-End Control


🔥 **[Check out our Project Page for more real-world demo videos and qualitative results!](https://xupeng23.github.io/AerialVLA)**

![Teaser](docs/assets/teaser.gif)

## 📖 Abstract
Vision-Language Navigation (VLN) for Unmanned Aerial Vehicles (UAVs) demands complex visual interpretation and continuous control in dynamic 3D environments. Existing hierarchical approaches rely on dense oracle guidance or auxiliary object detectors, creating semantic gaps and limiting genuine autonomy. We propose AerialVLA, a minimalist end-to-end Vision-Language-Action framework mapping raw visual observations and fuzzy linguistic instructions directly to continuous physical control signals. First, we introduce a streamlined dual-view perception strategy that reduces visual redundancy while preserving essential cues for forward navigation and precise grounding, which additionally facilitates future simulation-to-reality transfer. To reclaim genuine autonomy, we deploy a fuzzy directional prompting mechanism derived solely from onboard sensors, completely eliminating the dependency on dense oracle guidance. Ultimately, we formulate a unified control space that integrates continuous 3-Degree-of-Freedom (3-DoF) kinematic commands with an intrinsic landing signal, freeing the agent from external object detectors for precision landing. Extensive experiments on the TravelUAV benchmark demonstrate that AerialVLA achieves state-of-the-art performance in seen environments. Furthermore, it exhibits superior generalization in unseen scenarios by achieving nearly three times the success rate of leading baselines, validating that a minimalist, autonomy-centric paradigm captures more robust visual-motor representations than complex modular systems.

## 🚀 News & Updates

## 🛠️ Installation

## 📋 TODO List
We are continuously working on improving AerialVLA and pushing it towards real-world applications.
[ ] Release inference code and pre-trained weights.
[ ] Release the minimalist dual-view perception pipeline.
[ ] Hardware Deployment: Real-world UAV deployment guidelines and optimized inference code for Nvidia Jetson Orin edge computing platforms.

## 📄 License
This project is licensed under the Apache License 2.0. This permissive license supports both academic research and enterprise collaborations. See the LICENSE file for more details.

## ✒️ Citation

