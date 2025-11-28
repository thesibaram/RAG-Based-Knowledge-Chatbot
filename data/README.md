# Data Directory

This directory stores the datasets used by the Hospital Review RAG Chatbot.

- `raw/`: Contains the original CSV files (e.g., `reviews.csv`).

## Adding New Data

1. Place additional CSV files in the `raw/` directory.
2. Ensure each CSV has a `review` column containing text entries.
3. Update the application configuration if the file structure changes.

> **Note:** Large datasets should not be committed to version control. Add them to `.gitignore` if necessary.
