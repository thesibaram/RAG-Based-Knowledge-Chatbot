# Contributing to Hospital Review RAG Chatbot

Thank you for your interest in contributing! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/hospital-review-rag-chatbot.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use type hints for function arguments and return values
- Add docstrings to all functions, classes, and modules
- Keep functions focused and under 50 lines when possible

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in the present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Add a detailed description if necessary

### Testing

Before submitting a pull request:

1. Test your changes locally
2. Ensure the chatbot runs without errors: `python app.py`
3. Verify vector database creation: `python build_vectorstore.py`
4. Run evaluation if applicable: `python evaluate.py`

## Submitting Changes

1. Push to your fork: `git push origin feature/your-feature-name`
2. Open a Pull Request on GitHub
3. Describe your changes and their purpose
4. Link any related issues

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community

## Questions?

Feel free to open an issue if you have questions or need help!
