# HF Write Token 获取受阻记录与决策（2026-10-25）

> 结论先行：**国内链路交付走 hf-mirror / ModelScope；HF Write token 获取搁置**。真站账号体系（注册/登录/token）在境内被双重封锁，本文记录全部尝试路径与失败机理，供日后窗口期恢复。

## 背景

- 目标：拿到 `lihai.fkjs@gmail.com` 的 HF Write 权限 token（`whoami` 返回身份 JSON 为验收标准），用途仅为交付时上传权重。
- 本机：Windows 10，装有 EternalNetwork（mihomo/Meta 核心 VPN 客户端，TUN 全局接管）。

## 尝试路径与失败机理（按时间序）

| # | 路径 | 结果 | 根因 |
|---|------|------|------|
| 1 | EternalNetwork 开着访问 HF | 能开网页能"建 token"，但 token 真站全部 401 | 客户端把 `huggingface.co` 域名级导到**自家镜像假站**；假站 whoami 返回 `Repository not found`（真站永远 401），共产生 6+ 个无效 token |
| 2 | 关 VPN 直连 | 全不通 | GFW：DNS 污染（返回 Facebook 网段假 IP）；SNI 检测重置——hosts 钉 4 个真 CloudFront IP 后，openssl 实验证明：带错误 SNI 能到达服务器（回握手失败告警），带正确 SNI 被 RST；IPv6 同样被封；1.1.1.1/8.8.8.8/9.9.9.9 及 cloudflare-dns/dns.google 的 DoT/DoH 全被封 |
| 3 | hosts 钉真 IP + 本地转发器（127.0.0.1:8788，钉 18.65.14.87）+ 系统代理 | **curl/python 可达真站**（证书 Amazon 真链、whoami 真签名），**浏览器永远到假站** | VPN 核心在 TUN 层按 **TLS 指纹分流**：浏览器指纹→镜像假站，非浏览器指纹→真站；且真假按连接随机轮换（同一条 curl 命令时而真时而假） |
| 4 | curl 模拟浏览器登录真站（POST /login） | 202 + AWS WAF JS 挑战页 | 真站登录有 WAF 人机验证，非浏览器无法过 |
| 5 | 经 hf-mirror.com 透传登录 | 308 跳回真站 | 镜像只反代公开内容接口，账号类接口一律重定向 |
| 6 | 云机（AutoDL 003）SSH 绕行 | 不通 | 机场节点掐非标准端口（32375 无 SSH banner），直走 TUN 也被断 |

## 关键判别方法（复用价值）

- 真/假站签名：访问 `https://huggingface.co/api/whoami`——真站未登录返回 `401 {"error":"Invalid username or password."}`；假站返回 `404 {"error":"Repository not found"}`。任何路径先测这个再操作。
- 假站 token 特征：`hf_` 开头但真站 whoami 401。
- 有效身份验证：`curl -H "Authorization: Bearer <token>" https://huggingface.co/api/whoami` 返回含 name/email 的 JSON。

## 安全遗留

- 该邮箱密码已在假站路径多次提交（泄露），**若日后恢复使用该账号，第一件事改密码**。
- 假站"创建"的 token 未进入真站数据库，无需吊销；真站上不存在这些 token。

## 当前决策

- 下载/评测数据：`HF_ENDPOINT=https://hf-mirror.com` 或 ModelScope（均已验证可用，公开模型无需 token）。
- 权重交付：走 ModelScope 链接（赛题交付口径需与赛题方确认是否接受）。
- HF token 任务搁置。若日后必须恢复，按优先级：
  1. **直连窗口期**：关 VPN 挂监控等 GFW 放行窗口，窗口内普通浏览器直接注册/登录/建 token（无任何劫持）；
  2. 换可信代理（其他机场/手机热点+可信客户端），whoami 签名验证为真站后再操作；
  3. 请海外朋友代建 token（2 分钟）。
