# MCC

https://github.com/user-attachments/assets/f8fde795-6bf6-4978-8450-b4e5f075f6b4

`MCC` is a Unity project focused on **motion / character control**.

It is an official implementation of the original papers built on top of Unity and Sentis, intended to validate multiple motion-generation and character-control ideas within a shared runtime framework.

This project currently reproduces and integrates ideas from the following works:
- `Mode-adaptive neural networks for quadruped motion control`
- `DeepPhase: periodic autoencoders for learning motion phase manifolds`
- `Taming Diffusion Probabilistic Models for Character Control`

Within this project, these ideas correspond to three demos:
- `MANN`: motion generation and control based on mode-adaptive / trajectory-conditioned modeling.
- `DeepPhase`: motion generation with explicit phase-manifold / periodic latent representation.
- `CAMDM`: character control driven by diffusion probabilistic models.
