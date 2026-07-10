# lab-workspace 开发镜像：在 Ascend 基座上叠加 rustup/cargo（供 Probing 等）
# 默认基座用登录机已有的 a3-cann；可用 --build-arg BASE_IMAGE=... 换成 mindspeed
ARG BASE_IMAGE=registry2.d.pjlab.org.cn/lepton-trainingjob/a3-cann:8.3.rc2-a3-openeuler24.03-py3.11
FROM ${BASE_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:/opt/rust/rustup/bin:${PATH}

# 构建期代理（本机 Clash 经 SSH -R 打到登录机 127.0.0.1:18080）
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG NO_PROXY=localhost,127.0.0.1,registry2.d.pjlab.org.cn,.pjlab.org.cn
ARG no_proxy=localhost,127.0.0.1,registry2.d.pjlab.org.cn,.pjlab.org.cn
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    NO_PROXY=${NO_PROXY} \
    no_proxy=${no_proxy}

RUN pip install --no-cache-dir pyyaml maturin 2>/dev/null || \
    python3 -m pip install --no-cache-dir pyyaml maturin || true

# 预置本机下载的 aarch64 rustup-init（避免构建机直连 CDN 超时）
COPY rustup-init /tmp/rustup-init
RUN chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --default-toolchain stable --profile minimal \
    && rm -f /tmp/rustup-init \
    && rustc -V && cargo -V \
    && rustup component add rustfmt clippy 2>/dev/null || true

# 清掉镜像内代理环境，避免污染运行时
ENV HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy=

WORKDIR /workspace
CMD ["bash"]
