# Who Is Who in a Swarm? (simulation)

Code for the short report submitted for the Apart Research x AI Alignment Foundation Swarm Alignment Hackathon (Track 3), by Yethikrishna R, Mynd Labs, Thrissur, Kerala, India.

**This is a synthetic simulation, not an experiment on real language-model agents.** Parameters were chosen by hand and are not calibrated to any real incident.

It compares five ways of attributing agent actions (self-chosen name, operator attestation, per-agent signing key, behavioural fingerprint, key plus fingerprint) under renaming, key rotation, cloned agents and impersonators.

- `whoiswho_sim.py`: the simulator (NumPy only). `python3 whoiswho_sim.py 30` runs 30 seeds (1000 to 1029) for every sweep.
- `plots.py`: makes the figures and prints the tables from `results.json`.
- `results.json`: the results used in the report.

**Preexisting work:** none. All code here was written on 9 November 2026 (IST), after the sprint weekend and inside the submission window stated by the organisers. No earlier code was reused.

**AI use:** the simulator, analysis and report text were produced with an AI assistant acting for the author, who is responsible for the submission.

Limitations are listed in Appendix A of the report. Main ones: synthetic agents, hand-chosen parameters, impersonators do not imitate style, clones are near-identical by construction, enrolment is assumed honest.

The submitted 5-page PDF report is not in this repository. It was submitted to Apart through the sprint page.
