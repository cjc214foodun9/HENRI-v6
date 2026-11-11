
$ErrorActionPreference = "Continue"
$log = "verification_log.txt"
"Cycle S verification chain (relaunch) started $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
$commands = @(
  @("build_seed", "scripts/build_holarch_seed_cartridge.py"),
  @("j_cycle", "scripts/run_holarch_j_cycle.py --stages j1,j2,j3,j4,eval"),
  @("k_cycle", "scripts/run_holarch_k_cycle.py --stages k1,k2,k3,k4,eval"),
  @("m_cycle", "scripts/run_holarch_m_cycle.py --stages m1,m2,m3,m4,eval"),
  @("o_cycle", "scripts/run_holarch_o_cycle.py --stages o1,o2,o3,o4,eval"),
  @("p_cycle", "scripts/run_holarch_p_cycle.py --stages p1,p2,p3,p4,eval"),
  @("q_cycle", "scripts/run_holarch_q_cycle.py --stages q1,q2,q3,q4,eval"),
  @("r_cycle", "scripts/run_holarch_r_cycle.py --stages r1,r2,r3,r4,eval"),
  @("s_cycle", "scripts/run_holarch_s_cycle.py --stages s1,s2,s3,s4,eval"),
  @("aaii", "scripts/run_holarch_aaii_v5_eval.py"),
  @("agent_eval", "scripts/run_holarch_agent_eval.py")
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
"[pytest] start $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
& python -m pytest tests -q -k holarch *>&1 | Out-File -FilePath $log -Append -Encoding utf8
"RESULT pytest exit=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
"CHAIN DONE $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
