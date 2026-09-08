#!/usr/bin/env bash

###############################################################################
# NVIDIA + CUDA 13.3 + DCGM 4 + DKMS  (+ Broadcom bnxt_en quando aplicável)
#
# Ubuntu 24.04 LTS
#
# SCRIPT UNIFICADO — SUBSTITUI gpu_setup.sh e gpu_bnxt_setup.sh
# -----------------------------------------------------------------------------
# Este script roda em QUALQUER servidor (H100 / RTX6000 / etc.), com ou sem
# NIC Broadcom. Ele detecta automaticamente, via PCI (lspci), se o servidor
# possui uma NIC Broadcom:
#
#   - Se DETECTAR Broadcom  -> executa também toda a rotina específica de
#                              validação/gestão do driver bnxt_en (equivalente
#                              ao antigo gpu_bnxt_setup.sh).
#   - Se NÃO detectar        -> pula inteiramente essa rotina (equivalente ao
#                              antigo gpu_setup.sh), sem falhar por falta do
#                              módulo bnxt_en.
#
# A detecção pode ser forçada manualmente (bypass do auto-detect), exportando
# a variável de ambiente NIC_MODE antes de chamar o script:
#
#   sudo NIC_MODE=broadcom ./gpu_setup_auto.sh   # força rotina Broadcom
#   sudo NIC_MODE=generic  ./gpu_setup_auto.sh   # pula rotina Broadcom
#   sudo ./gpu_setup_auto.sh                     # NIC_MODE=auto (padrão)
#
# Todo o restante do fluxo (proteção de kernel, NVIDIA, CUDA, DCGM, nvtop,
# gpu-burn, validações finais) é comum a qualquer servidor e usa como base a
# versão mais rigorosa (antigo gpu_bnxt_setup.sh): checagens de Secure Boot,
# dpkg --audit, simulação de instalação para bloquear troca de kernel, e
# apt autoremove desabilitado (para não arriscar remover módulos DKMS ativos).
#
# OBJETIVOS
# -----------------------------------------------------------------------------
# - Preservar o kernel atualmente em execução
# - Não instalar outro kernel
# - Não atualizar/trocar o kernel ativo
# - Instalar exatamente os headers do kernel ativo
# - Instalar NVIDIA usando DKMS
# - Detectar automaticamente QUALQUER NÚMERO DE GPUs NVIDIA
# - Validar todas as GPUs NVIDIA via PCI x nvidia-smi
# - Instalar CUDA Toolkit 13.3
# - Instalar DCGM 4 para CUDA 13
# - Detectar automaticamente a NIC (Broadcom x genérica)
# - Preservar/validar Broadcom bnxt_en SOMENTE quando aplicável
# - Suportar bnxt-en-dkms (desabilitado por padrão para kernels 7.0+)
# - Não executar apt autoremove
# - Abortar antes do reboot se houver erro crítico
#
###############################################################################

set -Eeuo pipefail

###############################################################################
# CONFIGURAÇÃO
###############################################################################

EXPECTED_UBUNTU="24.04"

# Fonte única da versão do CUDA: mude MAJOR/MINOR aqui e todo o resto do
# script (nome do pacote, diretório de instalação, nvcc, profile.d) se ajusta
# automaticamente — nenhum outro lugar do script deve conter "13.3"/"13-3"
# literal.
CUDA_MAJOR="13"
CUDA_MINOR="3"

CUDA_VERSION_DASH="${CUDA_MAJOR}-${CUDA_MINOR}"   # formato de pacote: 13-3
CUDA_VERSION_DOT="${CUDA_MAJOR}.${CUDA_MINOR}"    # formato de path:    13.3

CUDA_TOOLKIT_PACKAGE="cuda-toolkit-${CUDA_VERSION_DASH}"
# DCGM: a NVIDIA versiona o Data Center GPU Manager de forma independente do
# CUDA. "4" é o major mais recente conhecido no momento em que este script foi
# escrito; é usado apenas como fallback caso a detecção dinâmica (feita mais
# adiante, após o repositório CUDA já estar configurado) não encontre nada no
# APT para o CUDA_MAJOR em uso.
DCGM_MAJOR_PINNED="4"
DCGM_PACKAGE="datacenter-gpu-manager-${DCGM_MAJOR_PINNED}-cuda${CUDA_MAJOR}"

CUDA_INSTALL_DIR="/usr/local/cuda-${CUDA_VERSION_DOT}"
CUDA_PROFILE_SCRIPT="/etc/profile.d/cuda-${CUDA_VERSION_DASH}.sh"
NVCC="${CUDA_INSTALL_DIR}/bin/nvcc"

# cuda-keyring: instala a chave GPG + fonte APT do repositório oficial da
# NVIDIA. Esta versão é FIXADA MANUALMENTE — não há como descobri-la de forma
# confiável em tempo de execução, pois é justamente o pacote necessário para
# começar a confiar no repositório (bootstrap). Diferente do CUDA/DCGM, isto
# não está relacionado à versão do Ubuntu nem à versão do CUDA Toolkit acima.
# Se o download abaixo falhar (404), confira a versão atual em:
#   https://developer.nvidia.com/cuda-downloads
#   (Linux > x86_64 > Ubuntu > 24.04 > deb (network))
# e atualize o valor abaixo.
CUDA_KEYRING_VERSION="1.1-1"
CUDA_KEYRING_PACKAGE="cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb"

CUDA_REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
CUDA_KEYRING_URL="${CUDA_REPO_BASE}/${CUDA_KEYRING_PACKAGE}"

# Broadcom:
# 0 = Usa o driver bnxt_en nativo do kernel (recomendado para Kernel 7.0+)
# 1 = Tenta instalar bnxt-en-dkms (pode falhar em kernels recentes)
ENABLE_BNXT_DKMS=0

# Ferramentas opcionais:
INSTALL_NVTOP=1
INSTALL_GPU_BURN=1

# Secure Boot:
# 0 = aborta se Secure Boot estiver habilitado
# 1 = continua mesmo com Secure Boot habilitado
ALLOW_SECURE_BOOT=0

# Reboot:
# 1 = reinicia automaticamente ao final
# 0 = não reinicia
AUTO_REBOOT=1

###############################################################################
# VARIÁVEIS
###############################################################################

CURRENT_KERNEL="$(uname -r)"
ARCH="$(dpkg --print-architecture)"

LOG_FILE="/var/log/nvidia-cuda-dcgm-install.log"

APT_OPTIONS=(
    -y
    -o Dpkg::Options::="--force-confdef"
    -o Dpkg::Options::="--force-confold"
)

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

###############################################################################
# LOG
###############################################################################

log_info() {
    echo -e "\e[34m[INFO]\e[0m $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "\e[32m[SUCESSO]\e[0m $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "\e[33m[AVISO]\e[0m $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "\e[31m[ERRO]\e[0m $*" | tee -a "$LOG_FILE" >&2
}

separator() {
    echo "===============================================================================" \
        | tee -a "$LOG_FILE"
}

###############################################################################
# ERROR HANDLER
###############################################################################

error_handler() {

    local exit_code=$?
    local line_number="$1"

    separator

    log_error "FALHA CRÍTICA."
    log_error "Linha       : $line_number"
    log_error "Exit code   : $exit_code"
    log_error "Kernel      : $CURRENT_KERNEL"
    log_error "Log         : $LOG_FILE"

    separator

    log_info "DKMS status:"
    dkms status 2>&1 | tee -a "$LOG_FILE" || true

    separator

    log_info "DPKG audit:"
    dpkg --audit 2>&1 | tee -a "$LOG_FILE" || true

    separator

    log_error "O sistema NÃO será reiniciado."

    exit "$exit_code"
}

trap 'error_handler $LINENO' ERR

###############################################################################
# ROOT
###############################################################################

if [[ "$EUID" -ne 0 ]]; then

    echo
    echo "[ERRO] Execute este script como root:"
    echo
    echo "sudo $0"
    echo

    exit 1
fi

###############################################################################
# LOG
###############################################################################

touch "$LOG_FILE"
chmod 600 "$LOG_FILE"

###############################################################################
# HEADER
###############################################################################

separator

log_info "NVIDIA / CUDA / DCGM / DKMS INSTALLER"
log_info "Ubuntu esperado : $EXPECTED_UBUNTU"
log_info "Kernel atual    : $CURRENT_KERNEL"
log_info "Arquitetura     : $ARCH"
log_info "Log             : $LOG_FILE"

separator

###############################################################################
# VALIDAR UBUNTU
###############################################################################

log_info "Validando sistema operacional..."

if [[ ! -r /etc/os-release ]]; then
    log_error "/etc/os-release não encontrado."
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" ]]; then
    log_error "Sistema não é Ubuntu."
    log_error "ID detectado: ${ID:-unknown}"
    exit 1
fi

if [[ "${VERSION_ID:-}" != "$EXPECTED_UBUNTU" ]]; then
    log_error "Versão Ubuntu incompatível."
    log_error "Detectado: ${VERSION_ID:-unknown}"
    log_error "Esperado : $EXPECTED_UBUNTU"
    exit 1
fi

log_success "Ubuntu 24.04 confirmado."

###############################################################################
# VALIDAR ARQUITETURA
###############################################################################

if [[ "$ARCH" != "amd64" ]]; then
    log_error "Arquitetura incompatível: $ARCH"
    log_error "CUDA NVIDIA x86_64 requer amd64 neste script."
    exit 1
fi

log_success "Arquitetura amd64 confirmada."

###############################################################################
# APT LOCK
###############################################################################

wait_for_apt_locks() {

    local waited=0
    local warning=0

    local locks=(
        /var/lib/dpkg/lock
        /var/lib/dpkg/lock-frontend
        /var/lib/apt/lists/lock
        /var/cache/apt/archives/lock
    )

    while fuser "${locks[@]}" >/dev/null 2>&1; do

        if [[ "$warning" -eq 0 ]]; then
            log_warn "APT/DPKG está ocupado."
            log_info "Aguardando liberação dos locks..."
            warning=1
        fi

        echo -ne \
            "\r\e[K[AGUARDANDO] APT ocupado - ${waited}s"

        sleep 2
        waited=$((waited + 2))

    done

    if [[ "$warning" -eq 1 ]]; then
        echo
        log_success "APT liberado após ${waited}s."
    fi
}

###############################################################################
# APT INSTALL
###############################################################################

apt_install() {

    wait_for_apt_locks

    log_info "apt-get install: $*"

    apt-get install \
        "${APT_OPTIONS[@]}" \
        "$@"
}

###############################################################################
# DPKG CHECK
###############################################################################

separator
log_info "Verificando estado do DPKG..."

DPKG_AUDIT="$(dpkg --audit || true)"

if [[ -n "$DPKG_AUDIT" ]]; then

    log_error "Existem pacotes quebrados ou configuração pendente:"
    echo "$DPKG_AUDIT" | tee -a "$LOG_FILE"

    log_error "O script NÃO executará automaticamente:"
    log_error "apt --fix-broken install"

    log_error "Isso é proposital para impedir que o APT instale"
    log_error "um kernel inesperado."

    exit 1
fi

log_success "DPKG está consistente."

###############################################################################
# SECURE BOOT
###############################################################################

separator
log_info "Verificando Secure Boot..."

SECURE_BOOT="unknown"

if command -v mokutil >/dev/null 2>&1; then

    if mokutil --sb-state 2>/dev/null \
        | grep -qi "SecureBoot enabled"; then

        SECURE_BOOT="enabled"

    elif mokutil --sb-state 2>/dev/null \
        | grep -qi "SecureBoot disabled"; then

        SECURE_BOOT="disabled"

    fi

fi

if [[ "$SECURE_BOOT" == "enabled" ]]; then

    log_warn "Secure Boot está habilitado."

    if [[ "$ALLOW_SECURE_BOOT" -ne 1 ]]; then

        log_error "Secure Boot habilitado pode impedir o carregamento"
        log_error "dos módulos NVIDIA/Broadcom compilados por DKMS."

        log_error "Instalação interrompida."

        log_info "Se você realmente precisa continuar:"
        log_info "edite ALLOW_SECURE_BOOT=1"

        exit 1
    fi

    log_warn "ALLOW_SECURE_BOOT=1."
    log_warn "Continuando mesmo com Secure Boot."

elif [[ "$SECURE_BOOT" == "disabled" ]]; then

    log_success "Secure Boot desabilitado."

else

    log_warn "Estado do Secure Boot não pôde ser determinado."

fi

###############################################################################
# BLACKLIST NOUVEAU
###############################################################################

separator
log_info "Configurando blacklist para o driver open-source nouveau..."

cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF

chmod 644 /etc/modprobe.d/blacklist-nouveau.conf
log_success "Blacklist para nouveau configurado."

###############################################################################
# APT UPDATE
###############################################################################

separator
log_info "Atualizando índices APT..."

wait_for_apt_locks
apt-get update

###############################################################################
# INSTALA FERRAMENTAS BÁSICAS
###############################################################################

separator
log_info "Instalando ferramentas básicas..."

apt_install \
    build-essential \
    dkms \
    ca-certificates \
    curl \
    wget \
    gnupg \
    pciutils \
    usbutils \
    lsb-release \
    ubuntu-drivers-common \
    mokutil \
    kmod \
    initramfs-tools \
    software-properties-common \
    alsa-utils \
    snapd

###############################################################################
# HEADERS EXATOS DO KERNEL
###############################################################################

separator

log_info "Verificando headers do kernel:"
log_info "$CURRENT_KERNEL"

if ! apt-cache show \
    "linux-headers-${CURRENT_KERNEL}" \
    >/dev/null 2>&1; then

    log_error "Headers não disponíveis:"
    log_error "linux-headers-${CURRENT_KERNEL}"

    log_error "O script NÃO instalará headers de outro kernel."

    exit 1
fi

apt_install \
    "linux-headers-${CURRENT_KERNEL}"

###############################################################################
# VALIDAR KERNEL BUILD
###############################################################################

KERNEL_BUILD="/lib/modules/${CURRENT_KERNEL}/build"

if [[ ! -f "$KERNEL_BUILD/Makefile" ]]; then

    log_error "Kernel build tree inválido:"
    log_error "$KERNEL_BUILD"

    exit 1
fi

log_success "Kernel build tree OK."

###############################################################################
# HOLD DOS METAPACOTES
###############################################################################

separator
log_info "Aplicando HOLD aos metapacotes do kernel..."

for pkg in \
    linux-generic \
    linux-image-generic \
    linux-headers-generic; do

    if dpkg-query \
        -W \
        -f='${Status}\n' \
        "$pkg" \
        2>/dev/null \
        | grep -q 'install ok installed'; then

        apt-mark hold "$pkg" >/dev/null

        log_success "HOLD: $pkg"

    else

        log_info "Não instalado: $pkg"

    fi

done

###############################################################################
# HOLD DOS PACOTES DO KERNEL ATUAL
###############################################################################

separator
log_info "Aplicando HOLD aos pacotes do kernel atual..."

CURRENT_KERNEL_PACKAGES=(
    "linux-image-${CURRENT_KERNEL}"
    "linux-headers-${CURRENT_KERNEL}"
    "linux-modules-${CURRENT_KERNEL}"
    "linux-modules-extra-${CURRENT_KERNEL}"
)

for pkg in "${CURRENT_KERNEL_PACKAGES[@]}"; do

    if dpkg-query \
        -W \
        -f='${Status}\n' \
        "$pkg" \
        2>/dev/null \
        | grep -q 'install ok installed'; then

        apt-mark hold "$pkg" >/dev/null

        log_success "HOLD: $pkg"

    else

        log_info "Não instalado: $pkg"

    fi

done

###############################################################################
# SALVA ESTADO DOS KERNELS
###############################################################################

separator
log_info "Registrando estado inicial dos kernels..."

dpkg-query \
    -W \
    -f='${Package}\t${Version}\t${Status}\n' \
    'linux-image*' \
    'linux-headers*' \
    'linux-modules*' \
    2>/dev/null \
    | tee -a "$LOG_FILE" \
    || true

###############################################################################
# CONFIGURA NVIDIA CUDA REPOSITORY
###############################################################################

separator
log_info "Configurando repositório oficial NVIDIA CUDA..."

TMP_KEYRING="/tmp/${CUDA_KEYRING_PACKAGE}"

rm -f "$TMP_KEYRING"

if ! curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 5 \
    --connect-timeout 15 \
    --max-time 60 \
    "$CUDA_KEYRING_URL" \
    --output "$TMP_KEYRING"; then

    log_error "Falha ao baixar $CUDA_KEYRING_PACKAGE de $CUDA_KEYRING_URL."
    log_error "A NVIDIA pode ter descontinuado esta versão do cuda-keyring."
    log_error "Verifique a versão atual em https://developer.nvidia.com/cuda-downloads"
    log_error "(Linux > x86_64 > Ubuntu > 24.04 > deb (network)) e atualize a variável"
    log_error "CUDA_KEYRING_VERSION no topo deste script."

    exit 1
fi

if [[ ! -s "$TMP_KEYRING" ]]; then
    log_error "Arquivo cuda-keyring vazio."
    exit 1
fi

dpkg \
    --install \
    --force-confdef \
    --force-confold \
    "$TMP_KEYRING"

rm -f "$TMP_KEYRING"

wait_for_apt_locks
apt-get update

log_success "Repositório NVIDIA configurado."

###############################################################################
# VERIFICA CUDA
###############################################################################

separator
log_info "Verificando $CUDA_TOOLKIT_PACKAGE..."

if ! apt-cache show \
    "$CUDA_TOOLKIT_PACKAGE" \
    >/dev/null 2>&1; then

    log_error "$CUDA_TOOLKIT_PACKAGE não está disponível."

    exit 1
fi

log_success "$CUDA_TOOLKIT_PACKAGE disponível."

###############################################################################
# DETECTA NVIDIA POR PCI ID E CONTA AS GPUs
###############################################################################

separator
log_info "Detectando GPUs NVIDIA por PCI ID..."

PCI_ALL="$(
    lspci -Dnn 2>/dev/null || true
)"

echo "$PCI_ALL" >> "$LOG_FILE"

NVIDIA_DEVICES="$(
    echo "$PCI_ALL" \
        | grep -Ei '10de:' \
        || true
)"

if [[ -z "$NVIDIA_DEVICES" ]]; then
    log_error "Nenhum dispositivo PCI com vendor NVIDIA (10de) encontrado."
    log_info "Saída completa do lspci:"
    echo "$PCI_ALL" | tee -a "$LOG_FILE"
    exit 1
fi

###############################################################################
# CONTA GPUs NVIDIA DINAMICAMENTE
###############################################################################

GPU_LINES="$(
    echo "$NVIDIA_DEVICES" \
        | grep -Ei 'VGA compatible controller|3D controller' \
        || true
)"

GPU_COUNT="$(echo "$GPU_LINES" | sed '/^[[:space:]]*$/d' | wc -l)"

if [[ "$GPU_COUNT" -eq 0 ]]; then
    log_error "Nenhum controlador de vídeo/3D NVIDIA encontrado."
    exit 1
fi

EXPECTED_GPU_COUNT="$GPU_COUNT"
log_success "GPUs NVIDIA detectadas via barramento PCI: $EXPECTED_GPU_COUNT"

###############################################################################
# MOSTRA MODELOS DE GPU
###############################################################################

log_info "Dispositivos de vídeo NVIDIA identificados:"
echo "$GPU_LINES" | tee -a "$LOG_FILE"

###############################################################################
# DETECTA DRIVER RECOMENDADO
###############################################################################

separator
log_info "Consultando driver NVIDIA recomendado..."

UBUNTU_DRIVER_OUTPUT="$(
    ubuntu-drivers devices 2>&1 || true
)"

echo "$UBUNTU_DRIVER_OUTPUT" >> "$LOG_FILE"

NVIDIA_DRIVER_PACKAGE="$(
    echo "$UBUNTU_DRIVER_OUTPUT" \
        | awk '
            /recommended/ {
                for (i=1; i<=NF; i++) {
                    if ($i ~ /^nvidia-driver-/) {
                        print $i
                        exit
                    }
                }
            }
        '
)"

if [[ -z "$NVIDIA_DRIVER_PACKAGE" ]]; then

    log_warn "Não foi possível identificar o driver recomendado via ubuntu-drivers."
    log_warn "Tentando fallback: maior nvidia-driver-N disponível no APT..."

    NVIDIA_DRIVER_PACKAGE="$(
        apt-cache pkgnames "nvidia-driver-" 2>/dev/null \
            | grep -E '^nvidia-driver-[0-9]+$' \
            | sort -t'-' -k3,3 -Vr \
            | head -1 \
            || true
    )"

    if [[ -z "$NVIDIA_DRIVER_PACKAGE" ]]; then

        log_error "Não foi possível determinar automaticamente qual driver NVIDIA instalar"
        log_error "(nem via ubuntu-drivers, nem via APT). Abortando em vez de assumir uma"
        log_error "versão de driver arbitrária/desatualizada. Verifique manualmente com"
        log_error "'ubuntu-drivers devices' e 'apt-cache pkgnames nvidia-driver-'."

        exit 1
    fi

    log_warn "Fallback selecionou dinamicamente: $NVIDIA_DRIVER_PACKAGE"

fi

log_success "Driver NVIDIA selecionado:"
log_success "$NVIDIA_DRIVER_PACKAGE"

###############################################################################
# DERIVA PACOTE NVIDIA DKMS
###############################################################################

NVIDIA_FLAVOR="${NVIDIA_DRIVER_PACKAGE#nvidia-driver-}"

NVIDIA_DKMS_PACKAGE="nvidia-dkms-${NVIDIA_FLAVOR}"

log_info "Pacote NVIDIA DKMS:"
log_info "$NVIDIA_DKMS_PACKAGE"

if ! apt-cache show \
    "$NVIDIA_DKMS_PACKAGE" \
    >/dev/null 2>&1; then

    log_error "Pacote DKMS NVIDIA não encontrado:"
    log_error "$NVIDIA_DKMS_PACKAGE"

    exit 1
fi

###############################################################################
# SIMULA NVIDIA DKMS E DRIVER
###############################################################################

separator
log_info "Simulando instalação do NVIDIA DKMS e Driver..."

NVIDIA_SIMULATION="$(
    apt-get \
        -s \
        install \
        "$NVIDIA_DKMS_PACKAGE" \
        "$NVIDIA_DRIVER_PACKAGE" \
        2>&1
)"

echo "$NVIDIA_SIMULATION" >> "$LOG_FILE"

###############################################################################
# BLOQUEIA NOVO KERNEL
###############################################################################

UNEXPECTED_KERNELS="$(
    echo "$NVIDIA_SIMULATION" \
        | grep '^Inst ' \
        | grep -E \
          'linux-(image|headers|modules|modules-extra)-[0-9]' \
        | grep -v -- "$CURRENT_KERNEL" \
        || true
)"

if [[ -n "$UNEXPECTED_KERNELS" ]]; then

    log_error "A instalação NVIDIA tentaria instalar outro kernel."

    echo "$UNEXPECTED_KERNELS" | tee -a "$LOG_FILE"

    log_error "Operação abortada."

    exit 1
fi

log_success "Simulação NVIDIA aprovada."

###############################################################################
# INSTALA NVIDIA DRIVER + DKMS (UNIFICADO)
###############################################################################

separator
log_info "Instalando NVIDIA DKMS e Driver em comando unificado..."

apt_install \
    "$NVIDIA_DKMS_PACKAGE" \
    "$NVIDIA_DRIVER_PACKAGE"

###############################################################################
# DKMS NVIDIA
###############################################################################

separator
log_info "Executando DKMS para kernel atual..."

dkms autoinstall \
    -k "$CURRENT_KERNEL" \
    || log_warn "DKMS autoinstall retornou erro (ignorado). Validando a seguir."

###############################################################################
# STATUS DKMS
###############################################################################

log_info "Status DKMS após NVIDIA:"

dkms status \
    | tee -a "$LOG_FILE"

###############################################################################
# VALIDA NVIDIA MODULE
###############################################################################

if ! modinfo \
    -k "$CURRENT_KERNEL" \
    nvidia \
    >/dev/null 2>&1; then

    log_error "Módulo NVIDIA não existe para $CURRENT_KERNEL."

    exit 1
fi

NVIDIA_MODULE_FILE="$(
    modinfo \
        -k "$CURRENT_KERNEL" \
        -F filename \
        nvidia
)"

log_success "NVIDIA module:"
log_success "$NVIDIA_MODULE_FILE"

###############################################################################
# DETECÇÃO DE NIC (BROADCOM x GENÉRICA)
###############################################################################

separator
log_info "Detectando tipo de NIC (Broadcom x genérica)..."

# NIC_MODE permite forçar manualmente, sem depender do auto-detect via PCI:
#   NIC_MODE=broadcom  -> força a rotina bnxt_en (mesmo sem detectar via PCI)
#   NIC_MODE=generic   -> pula a rotina bnxt_en (mesmo detectando Broadcom)
#   NIC_MODE=auto      -> (padrão) decide com base no lspci
NIC_MODE="${NIC_MODE:-auto}"

NIC_PCI_LINES="$(
    lspci -Dnn \
        | grep -Ei \
          'Ethernet controller|Network controller' \
        || true
)"

echo "$NIC_PCI_LINES" >> "$LOG_FILE"

HAS_BROADCOM_NIC=0

if echo "$NIC_PCI_LINES" | grep -Eqi 'Broadcom|14e4:'; then
    HAS_BROADCOM_NIC=1
fi

case "$NIC_MODE" in

    broadcom)
        HAS_BROADCOM_NIC=1
        log_warn "NIC_MODE=broadcom forçado manualmente (ignorando auto-detecção)."
        ;;

    generic)
        HAS_BROADCOM_NIC=0
        log_warn "NIC_MODE=generic forçado manualmente (ignorando auto-detecção)."
        ;;

    auto)
        : # mantém o resultado da auto-detecção acima
        ;;

    *)
        log_warn "NIC_MODE='$NIC_MODE' desconhecido. Usando auto-detecção."
        ;;

esac

if [[ -z "$NIC_PCI_LINES" ]]; then
    log_warn "Nenhum controlador de rede identificado via lspci."
fi

if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then

    log_success "NIC Broadcom detectada/selecionada."
    log_success "A rotina específica de bnxt_en SERÁ executada."
    echo "$NIC_PCI_LINES" \
        | grep -Ei 'Broadcom|14e4:' \
        | tee -a "$LOG_FILE" \
        || true

else

    log_info "Nenhuma NIC Broadcom detectada/selecionada."
    log_info "A rotina específica de bnxt_en será IGNORADA neste servidor."
    echo "$NIC_PCI_LINES" | tee -a "$LOG_FILE"

fi

###############################################################################
# BNXT MODULE
###############################################################################

if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then

    separator
    log_info "Validando bnxt_en..."

    if ! modinfo \
        -k "$CURRENT_KERNEL" \
        bnxt_en \
        >/dev/null 2>&1; then

        log_error "bnxt_en não existe para $CURRENT_KERNEL."

        exit 1
    fi

    BNXT_MODULE_FILE="$(
        modinfo \
            -k "$CURRENT_KERNEL" \
            -F filename \
            bnxt_en
    )"

    log_success "bnxt_en encontrado:"
    log_success "$BNXT_MODULE_FILE"

else

    separator
    log_info "NIC não-Broadcom: pulando validação do módulo bnxt_en."

fi

###############################################################################
# BNXT DKMS
###############################################################################

if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then

    if [[ "$ENABLE_BNXT_DKMS" -eq 1 ]]; then

        separator
        log_info "Verificando bnxt-en-dkms..."

        if dpkg-query \
            -W \
            -f='${Status}\n' \
            bnxt-en-dkms \
            2>/dev/null \
            | grep -q "install ok installed"; then

            BNXT_VERSION="$(
                dpkg-query \
                    -W \
                    -f='${Version}' \
                    bnxt-en-dkms
            )"

            log_success "bnxt-en-dkms instalado."
            log_info "Versão: $BNXT_VERSION"

            log_info "Reinstalando bnxt-en-dkms..."

            apt_install \
                --reinstall \
                bnxt-en-dkms

        elif apt-cache show \
            bnxt-en-dkms \
            >/dev/null 2>&1; then

            log_info "bnxt-en-dkms disponível no APT."

            apt_install \
                bnxt-en-dkms

        else

            log_warn "bnxt-en-dkms não está disponível."

            log_warn "Será utilizado o bnxt_en do kernel."

        fi

        ###########################################################################
        # DKMS após Broadcom
        ###########################################################################

        log_info "Executando DKMS novamente..."

        dkms autoinstall \
            -k "$CURRENT_KERNEL" \
            || log_warn "DKMS autoinstall retornou erro (ignorado). Validando a seguir."

    else

        separator
        log_info "Instalação do bnxt-en-dkms está DESABILITADA (ENABLE_BNXT_DKMS=0)."
        log_info "Utilizando driver nativo do kernel (in-tree)."

        if dpkg-query -W -f='${Status}\n' bnxt-en-dkms 2>/dev/null | grep -q "install ok installed"; then
            log_warn "O pacote bnxt-en-dkms está presente e incompatível com o kernel 7.0+."
            log_info "Removendo bnxt-en-dkms para limpar a árvore do DKMS..."
            apt-get remove --purge -y bnxt-en-dkms || true
        fi

    fi

else

    separator
    log_info "NIC não-Broadcom: pulando etapa de bnxt-en-dkms."

fi

###############################################################################
# VALIDA BNXT NOVAMENTE
###############################################################################

if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then

    if ! modinfo \
        -k "$CURRENT_KERNEL" \
        bnxt_en \
        >/dev/null 2>&1; then

        log_error "bnxt_en não está disponível."

        exit 1
    fi

    BNXT_FINAL_FILE="$(modinfo -k "$CURRENT_KERNEL" -F filename bnxt_en)"
    if [[ "$BNXT_FINAL_FILE" == *"/updates/dkms/"* ]]; then
        log_success "bnxt_en do DKMS está em uso: $BNXT_FINAL_FILE"
    else
        log_info "bnxt_en in-tree está em uso: $BNXT_FINAL_FILE"
    fi

fi

###############################################################################
# CUDA TOOLKIT
###############################################################################

separator
log_info "Instalando CUDA Toolkit ${CUDA_VERSION_DOT}..."

CUDA_SIMULATION="$(
    apt-get \
        -s \
        install \
        "$CUDA_TOOLKIT_PACKAGE" \
        2>&1
)"

echo "$CUDA_SIMULATION" >> "$LOG_FILE"

UNEXPECTED_CUDA_KERNELS="$(
    echo "$CUDA_SIMULATION" \
        | grep '^Inst ' \
        | grep -E \
          'linux-(image|headers|modules|modules-extra)-[0-9]' \
        | grep -v -- "$CURRENT_KERNEL" \
        || true
)"

if [[ -n "$UNEXPECTED_CUDA_KERNELS" ]]; then

    log_error "CUDA tentaria instalar outro kernel."

    echo "$UNEXPECTED_CUDA_KERNELS" \
        | tee -a "$LOG_FILE"

    exit 1
fi

apt_install \
    "$CUDA_TOOLKIT_PACKAGE"

###############################################################################
# CUDA PATH
###############################################################################

separator
log_info "Configurando CUDA ${CUDA_VERSION_DOT}..."

cat > "$CUDA_PROFILE_SCRIPT" <<EOF
export PATH=${CUDA_INSTALL_DIR}/bin:\${PATH}
export LD_LIBRARY_PATH=${CUDA_INSTALL_DIR}/lib64:\${LD_LIBRARY_PATH:-}
EOF

chmod 644 "$CUDA_PROFILE_SCRIPT"

###############################################################################
# NVCC
###############################################################################

if [[ ! -x "$NVCC" ]]; then

    log_error "nvcc não encontrado:"
    log_error "$NVCC"

    exit 1
fi

log_success "CUDA Toolkit ${CUDA_VERSION_DOT} instalado."

"$NVCC" \
    --version \
    | tee -a "$LOG_FILE"

###############################################################################
# DCGM
###############################################################################

separator
log_info "Detectando a major version do DCGM disponível no repositório..."

DCGM_MAJOR_DETECTED="$(
    apt-cache pkgnames "datacenter-gpu-manager-" 2>/dev/null \
        | grep -E "^datacenter-gpu-manager-[0-9]+-cuda${CUDA_MAJOR}$" \
        | grep -oP '(?<=datacenter-gpu-manager-)[0-9]+' \
        | sort -Vru \
        | head -1 \
        || true
)"

if [[ -n "$DCGM_MAJOR_DETECTED" ]]; then

    if [[ "$DCGM_MAJOR_DETECTED" != "$DCGM_MAJOR_PINNED" ]]; then
        log_warn "DCGM major disponível no repositório ($DCGM_MAJOR_DETECTED) difere do valor"
        log_warn "fixo no script ($DCGM_MAJOR_PINNED). Usando o valor detectado dinamicamente."
    fi

    DCGM_PACKAGE="datacenter-gpu-manager-${DCGM_MAJOR_DETECTED}-cuda${CUDA_MAJOR}"

else

    log_warn "Não foi possível detectar dinamicamente a major version do DCGM."
    log_warn "Usando o valor fixo do script: $DCGM_PACKAGE"

fi

log_info "Verificando DCGM..."

if ! apt-cache show \
    "$DCGM_PACKAGE" \
    >/dev/null 2>&1; then

    log_error "DCGM não disponível:"
    log_error "$DCGM_PACKAGE"

    exit 1
fi

log_success "DCGM disponível:"
log_success "$DCGM_PACKAGE"

###############################################################################
# DCGM INSTALL
###############################################################################

apt_install \
    --install-recommends \
    "$DCGM_PACKAGE"

###############################################################################
# DCGM SERVICE
###############################################################################

separator
log_info "Configurando serviço DCGM..."

if systemctl list-unit-files \
    | grep -q '^nvidia-dcgm\.service'; then

    systemctl enable nvidia-dcgm.service || true

elif systemctl list-unit-files \
    | grep -q '^dcgm\.service'; then

    systemctl enable dcgm.service || true

else

    log_warn "Serviço DCGM não encontrado."

fi

###############################################################################
# DEPMOD
###############################################################################

separator
log_info "Executando depmod..."

depmod \
    -a "$CURRENT_KERNEL"

###############################################################################
# INITRAMFS
###############################################################################

log_info "Atualizando initramfs somente do kernel atual..."

update-initramfs \
    -u \
    -k "$CURRENT_KERNEL"

###############################################################################
# VALIDAR KERNEL
###############################################################################

separator

if [[ "$(uname -r)" != "$CURRENT_KERNEL" ]]; then

    log_error "O kernel mudou durante a instalação."

    log_error "Inicial : $CURRENT_KERNEL"
    log_error "Atual   : $(uname -r)"

    exit 1
fi

log_success "Kernel permaneceu:"
log_success "$CURRENT_KERNEL"

###############################################################################
# NVIDIA MODULE LOAD
###############################################################################

separator
log_info "Tentando carregar módulo NVIDIA..."

if modprobe nvidia 2>>"$LOG_FILE"; then

    log_success "Módulo nvidia carregado."

else

    log_warn "Não foi possível carregar nvidia neste momento."
    log_warn "Será validado novamente após reboot."

fi

###############################################################################
# NVIDIA-SMI
###############################################################################

separator
log_info "Executando nvidia-smi..."

NVIDIA_SMI_COUNT=0

if command -v nvidia-smi >/dev/null 2>&1; then

    if nvidia-smi \
        | tee -a "$LOG_FILE"; then

        log_success "nvidia-smi respondeu corretamente."

    else

        log_warn "nvidia-smi não conseguiu acessar as GPUs."
        log_warn "Isso poderá ser resolvido após o reboot."

    fi

else

    log_error "nvidia-smi não foi instalado."

    exit 1

fi

###############################################################################
# NVIDIA-SMI -L
###############################################################################

separator
log_info "Enumerando GPUs NVIDIA via driver..."

NVIDIA_SMI_LIST="$(
    nvidia-smi -L 2>&1 || true
)"

echo "$NVIDIA_SMI_LIST" \
    | tee -a "$LOG_FILE"

NVIDIA_SMI_COUNT="$(
    echo "$NVIDIA_SMI_LIST" \
        | grep -c '^GPU ' \
        || true
)"

if [[ "$NVIDIA_SMI_COUNT" -eq "$EXPECTED_GPU_COUNT" ]]; then

    log_success "nvidia-smi detectou todas as $EXPECTED_GPU_COUNT GPUs."

elif [[ "$NVIDIA_SMI_COUNT" -eq 0 ]]; then

    log_warn "nvidia-smi ainda não detectou GPUs."
    log_warn "Isso será revalidado após reboot."

else

    log_warn "nvidia-smi detectou $NVIDIA_SMI_COUNT GPUs."
    log_warn "Esperado: $EXPECTED_GPU_COUNT"

fi

###############################################################################
# DCGM
###############################################################################

separator
log_info "Validando DCGM..."

if command -v dcgmi >/dev/null 2>&1; then

    dcgmi --version \
        | tee -a "$LOG_FILE" \
        || true

else

    log_warn "dcgmi não encontrado no PATH."

fi

###############################################################################
# SNAP
###############################################################################

separator
log_info "Configurando Snap..."

if command -v snap >/dev/null 2>&1; then

    systemctl enable --now snapd.socket || true

    snap wait system seed || true

    if [[ "$INSTALL_NVTOP" -eq 1 ]]; then

        log_info "Instalando nvtop..."

        if snap install nvtop; then
            log_success "nvtop instalado."
        else
            log_warn "Falha ao instalar nvtop."
        fi

    fi

    if [[ "$INSTALL_GPU_BURN" -eq 1 ]]; then

        log_info "Instalando gpu-burn..."

        if snap install gpu-burn; then
            log_success "gpu-burn instalado."
        else
            log_warn "Falha ao instalar gpu-burn."
        fi

    fi

else

    log_warn "Snap não está disponível."

fi

###############################################################################
# STATUS FINAL DKMS
###############################################################################

separator
log_info "STATUS FINAL DKMS"

dkms status \
    | tee -a "$LOG_FILE" \
    || true

###############################################################################
# STATUS BNXT
###############################################################################

separator
log_info "STATUS FINAL BNXT"

if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then

    modinfo \
        -k "$CURRENT_KERNEL" \
        bnxt_en \
        | grep -E \
          '^(filename|version|srcversion|vermagic|description):' \
        | tee -a "$LOG_FILE" \
        || true

else

    log_info "NIC não-Broadcom — bnxt_en não aplicável neste servidor."

fi

###############################################################################
# STATUS NVIDIA MODULE
###############################################################################

separator
log_info "STATUS FINAL NVIDIA MODULE"

modinfo \
    -k "$CURRENT_KERNEL" \
    nvidia \
    | grep -E \
      '^(filename|version|srcversion|vermagic|description):' \
    | tee -a "$LOG_FILE"

###############################################################################
# PCI NVIDIA
###############################################################################

separator
log_info "GPUs NVIDIA PCI"

lspci -Dnn \
    | grep -Ei \
      '10de:.*|NVIDIA.*3D controller|NVIDIA.*VGA' \
    | tee -a "$LOG_FILE"

###############################################################################
# PCI BROADCOM
###############################################################################

separator
log_info "Broadcom PCI (NIC de rede)"

lspci -Dnnk \
    | grep -A5 -B1 \
      -Ei 'Broadcom|14e4:' \
    | tee -a "$LOG_FILE" \
    || true

###############################################################################
# HOLDS
###############################################################################

separator
log_info "APT HOLDS"

apt-mark showhold \
    | tee -a "$LOG_FILE"

###############################################################################
# KERNELS
###############################################################################

separator
log_info "KERNELS INSTALADOS"

dpkg-query \
    -W \
    -f='${Package}\t${Version}\t${Status}\n' \
    'linux-image*' \
    'linux-headers*' \
    'linux-modules*' \
    2>/dev/null \
    | tee -a "$LOG_FILE" \
    || true

###############################################################################
# NÃO EXECUTAR AUTOREMOVE
###############################################################################

separator

log_warn "apt autoremove NÃO será executado."

log_info "Isso é proposital."

###############################################################################
# VALIDAÇÕES CRÍTICAS
###############################################################################

separator
log_info "VALIDAÇÕES CRÍTICAS FINAIS"

FINAL_ERRORS=0

###############################################################################
# KERNEL
###############################################################################

if [[ "$(uname -r)" == "$CURRENT_KERNEL" ]]; then

    log_success "[OK] Kernel: $CURRENT_KERNEL"

else

    log_error "[FAIL] Kernel mudou."
    FINAL_ERRORS=$((FINAL_ERRORS + 1))

fi

###############################################################################
# HEADERS
###############################################################################

if [[ -f "$KERNEL_BUILD/Makefile" ]]; then

    log_success "[OK] Kernel headers."

else

    log_error "[FAIL] Kernel headers."
    FINAL_ERRORS=$((FINAL_ERRORS + 1))

fi

###############################################################################
# NVIDIA MODULE
###############################################################################

if modinfo \
    -k "$CURRENT_KERNEL" \
    nvidia \
    >/dev/null 2>&1; then

    log_success "[OK] NVIDIA DKMS module."

else

    log_error "[FAIL] NVIDIA DKMS module."
    FINAL_ERRORS=$((FINAL_ERRORS + 1))

fi

###############################################################################
# BNXT
###############################################################################

if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then

    if modinfo \
        -k "$CURRENT_KERNEL" \
        bnxt_en \
        >/dev/null 2>&1; then

        log_success "[OK] bnxt_en."

    else

        log_error "[FAIL] bnxt_en."
        FINAL_ERRORS=$((FINAL_ERRORS + 1))

    fi

else

    log_info "[N/A] bnxt_en (NIC não-Broadcom, validação ignorada)."

fi

###############################################################################
# CUDA
###############################################################################

if [[ -x "$NVCC" ]]; then

    log_success "[OK] CUDA Toolkit ${CUDA_VERSION_DOT}."

else

    log_error "[FAIL] CUDA Toolkit ${CUDA_VERSION_DOT}."
    FINAL_ERRORS=$((FINAL_ERRORS + 1))

fi

###############################################################################
# DCGM
###############################################################################

if dpkg-query \
    -W \
    -f='${Status}\n' \
    "$DCGM_PACKAGE" \
    2>/dev/null \
    | grep -q "install ok installed"; then

    log_success "[OK] DCGM."

else

    log_error "[FAIL] DCGM."
    FINAL_ERRORS=$((FINAL_ERRORS + 1))

fi

###############################################################################
# NVIDIA GPU COUNT (PCI vs SMI)
###############################################################################

# Só comparamos se o nvidia-smi já tiver respondido, do contrário deixamos passar
# pois o módulo NVIDIA pode só estar disponível após o boot.
if [[ "$NVIDIA_SMI_COUNT" -gt 0 ]]; then
    if [[ "$NVIDIA_SMI_COUNT" -eq "$EXPECTED_GPU_COUNT" ]]; then
        log_success "[OK] GPUs detectadas no SO e no Driver: $EXPECTED_GPU_COUNT"
    else
        log_error "[FAIL] GPUs no SO (PCI): $EXPECTED_GPU_COUNT | GPUs no nvidia-smi: $NVIDIA_SMI_COUNT"
        FINAL_ERRORS=$((FINAL_ERRORS + 1))
    fi
else
    log_warn "[AVISO] Módulo/Driver nvidia pendente de reboot para listar GPUs."
fi

###############################################################################
# RESULTADO
###############################################################################

separator

if [[ "$FINAL_ERRORS" -ne 0 ]]; then

    log_error "=============================================="
    log_error " INSTALAÇÃO NÃO APROVADA"
    log_error "=============================================="

    log_error "Erros críticos: $FINAL_ERRORS"
    log_error "Kernel: $CURRENT_KERNEL"
    log_error "Log: $LOG_FILE"

    log_error "O servidor NÃO será reiniciado."

    exit 1
fi

###############################################################################
# SUCESSO
###############################################################################

log_success "=============================================="
log_success " INSTALAÇÃO APROVADA"
log_success "=============================================="

log_success "Ubuntu       : 24.04"
log_success "Kernel       : $CURRENT_KERNEL"
log_success "NVIDIA GPUs  : $EXPECTED_GPU_COUNT (Detectado)"
log_success "CUDA         : ${CUDA_VERSION_DOT}"
log_success "DCGM         : 4 / CUDA 13"
if [[ "$HAS_BROADCOM_NIC" -eq 1 ]]; then
    log_success "NIC          : Broadcom (bnxt_en OK)"
else
    log_success "NIC          : Não-Broadcom (bnxt_en N/A)"
fi
log_success "NVIDIA DKMS  : OK"
log_success "Log          : $LOG_FILE"

separator

###############################################################################
# REBOOT
###############################################################################

if [[ "$AUTO_REBOOT" -eq 1 ]]; then

    log_warn "O sistema será reiniciado em 10 segundos."
    log_warn "Pressione Ctrl+C para cancelar."

    sleep 10

    sync

    reboot

else

    log_warn "AUTO_REBOOT=0."
    log_warn "Nenhum reboot será executado."

fi
