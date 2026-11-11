
$ErrorActionPreference = "Continue"
$log = "final_log.txt"
"Final stage started $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
& python scripts/run_holarch_s_cycle.py --stages s1,s2,s3,s4,eval *>&1 | Out-File -FilePath $log -Append -Encoding utf8
"RESULT s_cycle exit=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
& python -m pytest tests -q -k holarch *>&1 | Out-File -FilePath $log -Append -Encoding utf8
"RESULT pytest exit=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
& python scratch_compose_docs.py *>&1 | Out-File -FilePath $log -Append -Encoding utf8
"RESULT docs exit=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
"FINAL DONE $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
