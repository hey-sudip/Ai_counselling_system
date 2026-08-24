from src.data.ingestion import DataIngestion


def main():
    ingestion = DataIngestion()

    paths = ingestion.ingest()

    print("\nReturned Paths")
    print("-" * 30)

    for key, value in paths.items():
        print(f"{key} : {value}")


if __name__ == "__main__":
    main()