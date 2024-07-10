import { gql } from 'graphql-request';

export const queryLabelSetIdsInVersion = gql`
query labelSetIdsInVersion($id: Int) {
  labelSetIdsInVersion(where: { versionId: { equals: $id } })
}
`;
export const queryLabelSet = gql`
  query LabelItems($id: Int) {
    labelSet(filter: { id: $id }) {
        imageId
        Image {
            ImageServer {
                hash
                name
                type
                url
            }
            rawPath
            path
            width
            height
        }
        LabelItem {
            id
            points
            LabelClass {
                name
            }
        }
    }
  }
`;