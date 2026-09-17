# README Image Assets

Place the paper's figures here as real, committed files (not expiring
`private-user-images.githubusercontent.com` links) so the README renders
reliably for anyone viewing it later:

- `fig1_architecture.png` — Fig. 1, overall system framework
- `fig2_gazebo_pipeline.png` — Fig. 2, Gazebo simulation pipeline
- `fig8_physical_testing.png` — Fig. 8, M30T physical experiment setup

Once added, reference them in the top-level `README.md` with relative
paths, e.g.:

```markdown
![System Architecture](docs/assets/fig1_architecture.png)
```

This avoids the JWT-signed, time-limited GitHub user-image URLs that will
403 once their embedded token expires.
