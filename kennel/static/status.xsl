<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="html" encoding="UTF-8" indent="yes"/>

<xsl:template match="/kennel">
<html lang="en">
<head>
  <meta http-equiv="refresh" content="2"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="/static/status.css"/>
  <title>kennel</title>
</head>
<body>
  <header>
    <h1><span class="paw">&#x1F43E;</span> kennel</h1>
  </header>
  <main>
    <xsl:choose>
      <xsl:when test="repo">
        <xsl:apply-templates select="repo"/>
      </xsl:when>
      <xsl:otherwise>
        <div class="empty">No repos configured. Napping &#x1F4A4;</div>
      </xsl:otherwise>
    </xsl:choose>
  </main>
</body>
</html>
</xsl:template>

<xsl:template match="repo">
<section>
  <xsl:attribute name="class">
    <xsl:text>repo</xsl:text>
    <xsl:if test="is_stuck = 'true'"> stuck</xsl:if>
    <xsl:if test="crash_count &gt; 0"> crashed</xsl:if>
    <xsl:if test="busy = 'true' and is_stuck != 'true'"> busy</xsl:if>
    <xsl:if test="busy != 'true' and is_stuck != 'true' and not(crash_count &gt; 0)"> idle</xsl:if>
  </xsl:attribute>

  <div class="repo-header">
    <span>
      <xsl:attribute name="class">
        <xsl:text>dot</xsl:text>
        <xsl:choose>
          <xsl:when test="is_stuck = 'true'"> dot-stuck</xsl:when>
          <xsl:when test="crash_count &gt; 0"> dot-crash</xsl:when>
          <xsl:when test="busy = 'true'"> dot-busy</xsl:when>
          <xsl:otherwise> dot-idle</xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>
    </span>
    <h2><xsl:value-of select="repo_name"/></h2>
    <xsl:if test="worker_uptime_seconds != ''">
      <span class="uptime">
        <xsl:text>up </xsl:text>
        <xsl:call-template name="format-duration">
          <xsl:with-param name="seconds" select="worker_uptime_seconds"/>
        </xsl:call-template>
      </span>
    </xsl:if>
  </div>

  <div class="activity">
    <xsl:value-of select="what"/>
  </div>

  <div class="meta">
    <xsl:if test="crash_count &gt; 0">
      <span class="badge badge-crash">
        <xsl:value-of select="crash_count"/>
        <xsl:text> crash</xsl:text>
        <xsl:if test="crash_count &gt; 1">es</xsl:if>
      </span>
    </xsl:if>

    <xsl:if test="rescoping = 'true'">
      <span class="badge badge-rescope">rescoping &#x27F3;</span>
    </xsl:if>

    <xsl:if test="session_alive = 'true'">
      <span class="badge badge-session">
        <xsl:text>session</xsl:text>
        <xsl:if test="session_owner != ''">
          <xsl:text>: </xsl:text>
          <xsl:value-of select="session_owner"/>
        </xsl:if>
        <xsl:if test="session_pid != ''">
          <xsl:text> (pid </xsl:text>
          <xsl:value-of select="session_pid"/>
          <xsl:text>)</xsl:text>
        </xsl:if>
      </span>
    </xsl:if>
  </div>

  <xsl:if test="claude_talker/kind">
    <div class="talker">
      <span class="talker-label">claude</span>
      <span class="talker-kind"><xsl:value-of select="claude_talker/kind"/></span>
      <span class="talker-desc"><xsl:value-of select="claude_talker/description"/></span>
      <xsl:if test="claude_talker/claude_pid != ''">
        <span class="talker-pid">
          <xsl:text>pid </xsl:text>
          <xsl:value-of select="claude_talker/claude_pid"/>
        </span>
      </xsl:if>
    </div>
  </xsl:if>

  <xsl:if test="last_crash_error != ''">
    <details class="crash-error">
      <summary>last crash</summary>
      <pre><xsl:value-of select="last_crash_error"/></pre>
    </details>
  </xsl:if>

  <xsl:if test="webhook_activities/webhook">
    <div class="webhooks">
      <span class="webhooks-label">webhooks</span>
      <ul>
        <xsl:for-each select="webhook_activities/webhook">
          <li>
            <xsl:value-of select="description"/>
            <span class="elapsed">
              <xsl:call-template name="format-duration">
                <xsl:with-param name="seconds" select="elapsed_seconds"/>
              </xsl:call-template>
            </span>
          </li>
        </xsl:for-each>
      </ul>
    </div>
  </xsl:if>

</section>
</xsl:template>

<!-- Format seconds into compact human-readable duration (matches CLI style) -->
<xsl:template name="format-duration">
  <xsl:param name="seconds"/>
  <xsl:variable name="s" select="floor(number($seconds))"/>
  <xsl:variable name="h" select="floor($s div 3600)"/>
  <xsl:variable name="m" select="floor(($s mod 3600) div 60)"/>
  <xsl:variable name="sec" select="$s mod 60"/>
  <xsl:choose>
    <xsl:when test="$h &gt; 0 and $m &gt; 0">
      <xsl:value-of select="$h"/>
      <xsl:text>h</xsl:text>
      <xsl:value-of select="$m"/>
      <xsl:text>m</xsl:text>
    </xsl:when>
    <xsl:when test="$h &gt; 0">
      <xsl:value-of select="$h"/>
      <xsl:text>h</xsl:text>
    </xsl:when>
    <xsl:when test="$m &gt; 0">
      <xsl:value-of select="$m"/>
      <xsl:text>m</xsl:text>
    </xsl:when>
    <xsl:otherwise>
      <xsl:value-of select="$sec"/>
      <xsl:text>s</xsl:text>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

</xsl:stylesheet>
