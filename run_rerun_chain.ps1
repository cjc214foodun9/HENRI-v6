
$ErrorActionPreference = "Continue"
$log = "rerun_log.txt"
"Rerun chain started $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
$commands = @(
  @("r_cycle", "scripts/run_holarch_r_cycle.py --stages r1,r2,r3,r4,eval"),
  @("s_cycle", "scripts/run_holarch_s_cycle.py --stages s1,s2,s3,s4,eval"),
  @("j_cycle", "scripts/run_holarch_j_cycle.py --stages j1,j2,j3,j4,eval"),
  @("k_cycle", "scripts/run_holarch_k_cycle.py --stages k1,k2,k3,k4,eval"),
  @("q_cycle", "scripts/run_holarch_q_cycle.py --stages q1,q2,q3,q4,eval")
)
foreach ($entry in $commands) {
  $name = $entry[0]
  $parts = $entry[1].Split(" ")
  $started = Get-Date
  "[$name] start $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
  if ($parts.Length -gt 1) {
    $rest = $parts[1..($parts.Length - 1)]
    & python $parts[0] @rest *>&1 | Out-File -FilePath $log -Append -Encoding utf8
  } else {
    & python $parts[0] *>&1 | Out-File -FilePath $log -Append -Encoding utf8
  }
  $code = $LASTEXITCODE
  $elapsed = ((Get-Date) - $started).TotalSeconds
  "RESULT $name exit=$code seconds=$([math]::Round($elapsed,1))" | Out-File -FilePath $log -Append -Encoding utf8
}
"RERUN DONE $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
