# Author:  Daniel Iwugo
# Comment: Christ is King
#
# Homebrew formula for the checker, and deliberately NOT for the whole tool.
#
# `senbonzakura` itself pulls torch and sixty-odd packages at roughly 5.8 GB. A formula would have
# to pin every one of them as a resource, and it would be out of date within a week. pip, Docker
# and Colab are the right homes for that.
#
# The checker is a different animal: it imports nothing outside the standard library, which is a
# measured claim rather than a hoped-for one. So it installs in seconds, works offline, and is the
# piece worth putting one command away from somebody.
#
# NO TAP REPOSITORY IS NEEDED. `brew tap` insists on a repository called `homebrew-<something>`,
# which would mean a second repo for one file. Installing from a URL does not:
#
#     brew install --formula \
#       https://raw.githubusercontent.com/elementmerc/senbonzakura/main/packaging/senbonzakura-check.rb
#
# The url and sha256 below are rewritten by `.github/workflows/distribute.yml` on a stable
# release, so this file is never hand-edited and never drifts from what was actually published.
class SenbonzakuraCheck < Formula
  include Language::Python::Virtualenv

  desc "Read an evaluation result file and report how the number could be wrong"
  homepage "https://github.com/elementmerc/senbonzakura"
  # PLACEHOLDER-URL
  url "https://github.com/elementmerc/senbonzakura/archive/refs/tags/v0.0.0.tar.gz"
  # PLACEHOLDER-SHA256
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "AGPL-3.0-or-later"

  depends_on "python@3.12"

  def install
    # The checker is a subdirectory of the repository rather than its own distribution, so the
    # build happens there. No resources block, because there are no dependencies to pin: that is
    # the whole reason this formula is viable and the one for the main tool is not.
    cd "checker" do
      virtualenv_install_with_resources
    end
  end

  test do
    # Not `--version`, which proves only that the script exists. This asks it to do its job on a
    # file it has never seen and checks that it reports rather than crashes.
    (testpath/"result.json").write <<~JSON
      {"results": {"demo": {"acc": 0.5}}, "n-samples": {"demo": {"effective": 3}}}
    JSON
    output = shell_output("#{bin}/senbonzakura-check #{testpath}/result.json 2>&1", 1)
    assert_match(/sample|small|3/i, output)
  end
end
