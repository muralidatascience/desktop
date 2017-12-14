import * as React from 'react'

import { Octicon, OcticonSymbol } from '../octicons'

import { ReleaseNote, ReleaseSummary } from '../../models/release-notes'

import { updateStore } from '../lib/update-store'
import { ButtonGroup } from '../lib/button-group'
import { Button } from '../lib/button'

import { Dialog, DialogContent, DialogFooter } from '../dialog'

interface IReleaseNotesProps {
  readonly onDismissed: () => void
  readonly newRelease: ReleaseSummary
}

/**
 * The dialog to show with details about the newest release
 */
export class ReleaseNotes extends React.Component<IReleaseNotesProps, {}> {
  private onCloseButtonClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (this.props.onDismissed) {
      this.props.onDismissed()
    }
  }

  private renderCloseButton() {
    // We're intentionally using <a> here instead of <button> because
    // we can't prevent chromium from giving it focus when the the dialog
    // appears. Setting tabindex to -1 doesn't work. This might be a bug,
    // I don't know and we may want to revisit it at some point but for
    // now an anchor will have to do.
    return (
      <a className="close" onClick={this.onCloseButtonClick}>
        <Octicon symbol={OcticonSymbol.x} />
      </a>
    )
  }

  private renderList(
    releaseEntries: ReadonlyArray<ReleaseNote>,
    header: string
  ): JSX.Element | null {
    if (releaseEntries.length === 0) {
      return null
    }

    const options = new Array<JSX.Element>()

    for (const [i, entry] of releaseEntries.entries()) {
      options.push(<li key={i}>{entry.message}</li>)
    }

    return (
      <div className="section">
        <p className="header">{header}</p>
        <ul className="entries">{options}</ul>
      </div>
    )
  }

  private drawSingleColumnLayout(release: ReleaseSummary): JSX.Element {
    return (
      <div className="container">
        {this.renderList(release.bugfixes, 'Bugfixes')}
        {this.renderList(release.enhancements, 'Enhancements')}
        {this.renderList(release.other, 'Other')}
      </div>
    )
  }

  private drawTwoColumnLayout(release: ReleaseSummary): JSX.Element {
    return (
      <div className="container">
        <div className="column">
          {this.renderList(release.enhancements, 'Enhancements')}
          {this.renderList(release.other, 'Other')}
        </div>
        <div className="column">
          {this.renderList(release.bugfixes, 'Bugfixes')}
        </div>
      </div>
    )
  }

  public render() {
    const release = this.props.newRelease

    const contents =
      release.enhancements.length > 0 && release.bugfixes.length > 0
        ? this.drawTwoColumnLayout(release)
        : this.drawSingleColumnLayout(release)

    return (
      <Dialog id="release-notes" onDismissed={this.props.onDismissed}>
        <DialogContent>
          <header className="dialog-header">
            <div className="title">
              <p className="version">Version {release.latestVersion}</p>
              <p className="date">{release.datePublished}</p>
            </div>
            {this.renderCloseButton()}
          </header>
          {contents}
        </DialogContent>
        <DialogFooter>
          <ButtonGroup destructive={true}>
            <Button type="submit">Close</Button>
            <Button onClick={this.updateNow}>
              {__DARWIN__ ? 'Install Now' : 'Install now'}
            </Button>
          </ButtonGroup>
        </DialogFooter>
      </Dialog>
    )
  }

  private updateNow = () => {
    updateStore.quitAndInstallUpdate()
  }
}