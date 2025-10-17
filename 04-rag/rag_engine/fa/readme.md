# RAG Engine — Persian Track
این پوشه شامل **مستندات فارسی** است. کدهای مشترک در `rag_engine/` باقی می‌مانند و منطق زبان‌محور فارسی در `rag_engine/fa/chunking.py` قرار دارد (در صورت نصب `hazm` کیفیت بخش‌بندی جملات بهتر می‌شود).

## شروع سریع

**ایندکس‌کردن PDF با ردّ حرکت (verbose)**

```bash
python cli.py index --pdf ./samples/fa_handbook.pdf --lang fa --out ./fa_index.pkl --chunk-size 400 --chunk-overlap 80 --emb tfidf --verbose
```

**پرسش (حالت آفلاین)**

```bash
python cli.py query --index ./fa_index.pkl --q " عوامل موثر بر تأثیر بازاریابی گوشه ای بر وفاداری مشتریان در صنایع ورزشی با استفاده از تکنیکهای هوش مصنوعی چیست؟" --top-k 4 --verbose
```

**پرسش با LLM (روتر سازگار با OpenAI)**

```bash
python cli.py query --index ./fa_index.pkl --q "answer this question and give result in structured format: عوامل موثر بر تأثیر بازاریابی گوشه ای بر وفاداری مشتریان در صنایع ورزشی با استفاده از تکنیکهای هوش مصنوعی چیست؟" --top-k 4 --llm --model gpt-4o-mini --base-url https://api.metisai.ir/openai/v1 --verbose
```

## نحوه‌ی مسیریابی زبان

* ماژول `rag_engine/chunking.py` وقتی `--lang fa` باشد به `rag_engine/fa/chunking.py` ارجاع می‌دهد.
* مسیر انگلیسی در `rag_engine/en/chunking.py` قرار دارد. ماژول‌های مشترک (CLI، ایندکسر، بازیاب، QA) در ریشه‌ی پکیج هستند.
