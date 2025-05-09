import os
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from classify import process_entries

app = FastAPI()

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    # 1. Ensure CSV upload
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        # 2. Read CSV and validate columns
        df = pd.read_csv(file.file)
        required = {"source", "log_message"}
        if not required.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {', '.join(required)}"
            )

        # 3. Perform classification using the renamed function
        entries = list(zip(df["source"], df["log_message"]))
        df["predicted_label"] = process_entries(entries)

        # 4. Persist to a unique file name (avoid overwriting concurrent requests)
        os.makedirs("resources", exist_ok=True)
        output_path = os.path.join("resources", f"output_{os.getpid()}.csv")
        df.to_csv(output_path, index=False)

        # 5. Return the new CSV
        return FileResponse(output_path, media_type="text/csv", filename="classified.csv")

    except HTTPException:
        # pass-through known HTTPExceptions
        raise
    except Exception as exc:
        # catch-all for unexpected
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        file.file.close()
