## Retraining and Monitoring Pipeline

Automated ML pipeline integrated with Github actions, Dagshub and DVC

## Structure

```
├── .github/workflows/
│   └── retrain-on-push.yaml   # Auto-retraining workflow
├── src/
│   ├── model.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── monitor.py             # Drift & performance monitoring
│   └── preprocess_new_data.py # Preprocess new data
├── data/
│   └── new_data.csv           # Upload new data here to trigger retraining
├── train/
│   └── train.csv              # Training data (DVC tracked)
├── test/
│   └── test.csv               # Test data (DVC tracked)
├── artifacts/                 # Centralised artifact storage (DVC tracked)
├── models/
│   └── model.keras            # Trained model (DVC tracked)
├── dvc.yaml                   # DVC pipeline definition
└── params.yaml                # Hyperparameters
...
```

## Trigers

The pipeline can be triggered when:
* New data is pushed 
* Mannually through github actions
* Automatically every 12 hours

## Pipeline Process

When any one one of these criteria is met, the pipeline runs as follows:
1. Set up job - Sets up Ubuntu virtual machine, downloads required action repositories, and sets job name.
2. Checkout code - Downloads the latest version of the repository onto the Ubuntu virtual machine 
3. Setup Python - Installs python version 3.11 onto the virtual machine
4. Configure Git user for DVCLive - Configures Git user so that DVC can make commits as the pipeline runs
5. Install dependancies - Installs all python libraries required for the pipline
6. Configure DVC remote - Connects DVC to dagshub remote storage using stored Github secrets to enable pulling and pushing of data.
7. Pull latest data and models - Downloads latest data, model, and artefacts files from Dagshub
8. Check for new data - Checks if new data has been created since the last time that the pipeline ran
9. Preprocess new data - Conducts preporcessing of new data, such as scaling
10. Run monitoring (for logging only) - compares model performance on new data against baseline performance
11. Retrain model (only if new data exists) - combines new and existing data, and retrains the model
12. Evaluate new model - evaluates model performance
13. Push new model and artefacts - uploads the retrained model and artefacts to dagshub
14. Commit and push changes - commits all updated files to the Github repository
15. Upload reports - saves training curves, evaluation netrics, and summaries of monitoring as downloadable artefacts on the Github actions run
16. Send notification - prints a confirmation of successful pipeline completion
17. Post setup python - Cleanup step
18. Post checkout code -  Cleanup step
19. Complete job - Job is marked as finished
 
## Commit Convention

This project uses semantic commits:
* `feat:` - new feature or data
* `fix:` - bug fix
* `chore:` - maintenance tasks
* `docs:` - documentation updates