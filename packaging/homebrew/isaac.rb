# I.S.A.A.C. Homebrew Formula

class Isaac < Formula
  desc "Intelligent System for Autonomous Action and Cognition — neuro-symbolic autonomous agent"
  homepage "https://github.com/Matheussoranco/I.S.A.A.C"
  license "Apache-2.0"
  version "1.6.2"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/Matheussoranco/I.S.A.A.C/releases/download/v#{version}/isaac-agent-#{version}-macos-arm64.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256_ARM64"
    else
      url "https://github.com/Matheussoranco/I.S.A.A.C/releases/download/v#{version}/isaac-agent-#{version}-macos-x86_64.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256_X86_64"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/Matheussoranco/I.S.A.A.C/releases/download/v#{version}/isaac-agent-#{version}-linux-x86_64.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256_LINUX_X86_64"
    end
  end

  depends_on "python@3.12"

  def install
    libexec.install Dir["*"]
    bin.install_symlink libexec/"bin/isaac" => "isaac"
  end

  test do
    system "#{bin}/isaac", "--version"
  end
end